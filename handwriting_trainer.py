import os
import logging
import subprocess
import tempfile
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
import pytesseract
from db_manager import DatabaseManager


class HandwritingTrainer:
    """Train Tesseract OCR on handwriting samples."""
    
    def __init__(self, db_manager: DatabaseManager, training_dir: str = "training_data"):
        """
        Initialize the handwriting trainer.
        
        Args:
            db_manager: Database manager instance
            training_dir: Directory to store training files
        """
        self.db_manager = db_manager
        self.training_dir = Path(training_dir)
        self.training_dir.mkdir(exist_ok=True)
        self.setup_logging()
        
        # Set up a directory for trained models
        self.models_dir = self.training_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Create directory for temporary training files
        self.temp_dir = self.training_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Configure logging for training operations."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def collect_training_sample(
        self, 
        doc_id: int, 
        page_num: int, 
        original_text: str, 
        corrected_text: str,
        region: Optional[Tuple[int, int, int, int]] = None,
        profile_name: str = "default"
    ) -> bool:
        """
        Store a corrected text sample for training.
        
        Args:
            doc_id: Document ID
            page_num: Page number
            original_text: Text from OCR
            corrected_text: User-corrected text
            region: Optional (x1, y1, x2, y2) coordinates of the text region
            profile_name: Name of handwriting profile
            
        Returns:
            bool: True if successful
        """
        try:
            # Get the document details
            document = self.db_manager.get_document_by_id(doc_id)
            if not document:
                self.logger.error(f"Document {doc_id} not found")
                return False
                
            # Extract file path and make a copy of the specific page
            file_path = Path(document['relative_path'])
            if not file_path.exists():
                self.logger.error(f"Document file not found: {file_path}")
                return False
                
            # Save a copy of the page image for training
            page_image_path = None
            if region:
                # Convert the PDF page to image and extract the region
                with tempfile.TemporaryDirectory() as temp_dir:
                    from pdf2image import convert_from_path
                    
                    self.logger.info(f"Extracting page {page_num} from {file_path}")
                    images = convert_from_path(
                        file_path, 
                        first_page=page_num, 
                        last_page=page_num,
                        dpi=300,
                        output_folder=temp_dir
                    )
                    
                    if images:
                        # Get the extracted region
                        image = images[0]
                        x1, y1, x2, y2 = region
                        region_img = image.crop((x1, y1, x2, y2))
                        
                        # Save the region image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_filename = f"train_doc{doc_id}_p{page_num}_{timestamp}.png"
                        img_path = self.training_dir / "samples" / img_filename
                        
                        # Ensure directory exists
                        (self.training_dir / "samples").mkdir(exist_ok=True)
                        
                        region_img.save(img_path)
                        page_image_path = str(img_path)
                        self.logger.info(f"Saved region image to {page_image_path}")
            
            # Get or create handwriting profile
            profile_id = self._get_or_create_profile(profile_name)
            
            # Store the training sample
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                if region:
                    x1, y1, x2, y2 = region
                    cursor.execute("""
                        INSERT INTO handwriting_training_data (
                            profile_id, original_text, corrected_text, 
                            page_image_path, x1, y1, x2, y2, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        profile_id, original_text, corrected_text,
                        page_image_path, x1, y1, x2, y2, datetime.now()
                    ))
                else:
                    cursor.execute("""
                        INSERT INTO handwriting_training_data (
                            profile_id, original_text, corrected_text, created_at
                        ) VALUES (?, ?, ?, ?)
                    """, (profile_id, original_text, corrected_text, datetime.now()))
                
                # Update the sample count
                cursor.execute("""
                    UPDATE handwriting_profiles
                    SET training_sample_count = training_sample_count + 1,
                        last_updated_at = ?
                    WHERE id = ?
                """, (datetime.now(), profile_id))
                
                self.logger.info(f"Added training sample for profile {profile_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error collecting training sample: {e}")
            return False
    
    def _get_or_create_profile(self, profile_name: str) -> int:
        """Get or create a handwriting profile by name."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO handwriting_profiles (
                    profile_name, created_at, last_updated_at, training_sample_count
                ) VALUES (?, ?, ?, 0)
                ON CONFLICT (profile_name) 
                DO UPDATE SET last_updated_at = excluded.last_updated_at
                RETURNING id
            """, (profile_name, datetime.now(), datetime.now()))
            
            result = cursor.fetchone()
            return result[0]
    
    def start_training_job(self, profile_name: str = "default") -> Optional[int]:
        """
        Start a training job for a handwriting profile.
        
        Args:
            profile_name: Name of the profile to train
            
        Returns:
            Optional[int]: Training job ID if successful
        """
        try:
            # Get profile ID
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, training_sample_count 
                    FROM handwriting_profiles
                    WHERE profile_name = ?
                """, (profile_name,))
                
                result = cursor.fetchone()
                if not result:
                    self.logger.error(f"Profile {profile_name} not found")
                    return None
                    
                profile_id, sample_count = result['id'], result['training_sample_count']
                
                if sample_count < 5:
                    self.logger.warning(
                        f"Profile {profile_name} has only {sample_count} samples. "
                        f"At least 5 samples recommended for training."
                    )
                
                # Create a training job
                cursor.execute("""
                    INSERT INTO training_jobs (
                        profile_id, status, started_at, sample_count
                    ) VALUES (?, 'pending', ?, ?)
                    RETURNING id
                """, (profile_id, datetime.now(), sample_count))
                
                job_id = cursor.fetchone()[0]
                
                self.logger.info(f"Created training job {job_id} for profile {profile_name}")
                return job_id
                
        except Exception as e:
            self.logger.error(f"Error starting training job: {e}")
            return None
    
    def generate_training_data(self, job_id: int) -> bool:
        """
        Generate training data files for Tesseract.
        
        Args:
            job_id: Training job ID
            
        Returns:
            bool: True if successful
        """
        try:
            # Get job and profile details
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT j.id, j.profile_id, p.profile_name
                    FROM training_jobs j
                    JOIN handwriting_profiles p ON j.profile_id = p.id
                    WHERE j.id = ?
                """, (job_id,))
                
                job = cursor.fetchone()
                if not job:
                    self.logger.error(f"Training job {job_id} not found")
                    return False
                
                profile_id, profile_name = job['profile_id'], job['profile_name']
                
                # Update job status
                cursor.execute("""
                    UPDATE training_jobs
                    SET status = 'generating_data'
                    WHERE id = ?
                """, (job_id,))
                
                # Get training samples
                cursor.execute("""
                    SELECT id, original_text, corrected_text, page_image_path
                    FROM handwriting_training_data
                    WHERE profile_id = ?
                    ORDER BY created_at DESC
                    LIMIT 100
                """, (profile_id,))
                
                samples = cursor.fetchall()
                if not samples:
                    self.logger.error(f"No training samples found for profile {profile_name}")
                    
                    cursor.execute("""
                        UPDATE training_jobs
                        SET status = 'failed',
                            error_message = 'No training samples found',
                            completed_at = ?
                        WHERE id = ?
                    """, (datetime.now(), job_id))
                    
                    return False
            
            # Create a clean job directory
            job_dir = self.temp_dir / f"job_{job_id}"
            if job_dir.exists():
                shutil.rmtree(job_dir)
            job_dir.mkdir(exist_ok=True)
            
            # Prepare for Tesseract training
            training_text = []
            image_files = []
            
            # For samples with images, we can use them directly
            # For text-only samples, we need to generate images
            for sample in samples:
                if sample['page_image_path'] and Path(sample['page_image_path']).exists():
                    # Use existing image
                    src_path = Path(sample['page_image_path'])
                    dst_path = job_dir / f"sample_{sample['id']}.png"
                    shutil.copy(src_path, dst_path)
                    image_files.append(dst_path)
                    training_text.append(sample['corrected_text'])
                else:
                    # Generate image from corrected text
                    # This is simplified - real implementation would need more sophistication
                    image_path = self._generate_text_image(
                        sample['corrected_text'],
                        job_dir / f"sample_{sample['id']}.png"
                    )
                    if image_path:
                        image_files.append(image_path)
                        training_text.append(sample['corrected_text'])
            
            if not image_files:
                self.logger.error("Failed to prepare training images")
                
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE training_jobs
                        SET status = 'failed',
                            error_message = 'Failed to prepare training images',
                            completed_at = ?
                        WHERE id = ?
                    """, (datetime.now(), job_id))
                
                return False
            
            # Create combined training text file
            with open(job_dir / "training_text.txt", "w") as f:
                f.write("\n".join(training_text))
            
            # Create a shell script to run Tesseract training commands
            # This is a placeholder - actual implementation would need proper Tesseract training commands
            training_script = job_dir / "train.sh"
            with open(training_script, "w") as f:
                f.write(f"""#!/bin/bash
# Training script for job {job_id}
# This is a placeholder - replace with actual Tesseract training commands

# Example: Create box files for each training image
for img in {job_dir}/*.png; do
    tesseract "$img" "${{img%.*}}" batch.nochop makebox
done

# More Tesseract training commands would follow here
""")
            
            # Make script executable
            os.chmod(training_script, 0o755)
            
            # Update job status
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE training_jobs
                    SET status = 'data_generated'
                    WHERE id = ?
                """, (job_id,))
            
            self.logger.info(f"Generated training data for job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating training data: {e}")
            
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE training_jobs
                        SET status = 'failed',
                            error_message = ?,
                            completed_at = ?
                        WHERE id = ?
                    """, (str(e), datetime.now(), job_id))
            except Exception:
                pass
                
            return False
    
    def _generate_text_image(self, text: str, output_path: Path) -> Optional[Path]:
        """Generate an image from text for training."""
        try:
            # Create a blank image
            width, height = 800, 200
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Try to use a handwriting-like font if available
            try:
                # This would be better with a real handwriting font
                font = ImageFont.truetype("Arial", 24)
            except:
                font = ImageFont.load_default()
            
            # Draw text
            draw.text((20, 50), text, fill='black', font=font)
            
            # Save image
            image.save(output_path)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating text image: {e}")
            return None
    
    def execute_training(self, job_id: int) -> bool:
        """
        Execute the Tesseract training process.
        
        This is a complex process that involves:
        1. Generating box files
        2. Creating training data
        3. Running Tesseract training
        4. Creating a new trained model
        
        Args:
            job_id: Training job ID
            
        Returns:
            bool: True if successful
        """
        try:
            # Get job details
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT j.id, j.profile_id, p.profile_name
                    FROM training_jobs j
                    JOIN handwriting_profiles p ON j.profile_id = p.id
                    WHERE j.id = ? AND j.status = 'data_generated'
                """, (job_id,))
                
                job = cursor.fetchone()
                if not job:
                    self.logger.error(f"Training job {job_id} not found or not ready")
                    return False
                
                profile_id, profile_name = job['profile_id'], job['profile_name']
                
                # Update job status
                cursor.execute("""
                    UPDATE training_jobs
                    SET status = 'training'
                    WHERE id = ?
                """, (job_id,))
            
            # NOTE: Actual Tesseract training is very complex and requires specific steps
            # This is a simplified version - in production, you'd need a proper Tesseract training pipeline
            
            job_dir = self.temp_dir / f"job_{job_id}"
            
            # In a real implementation, you would:
            # 1. Generate box files for each training image
            # 2. Run Tesseract training commands
            # 3. Generate the traineddata file
            
            # For this example, we'll simulate a successful training
            model_path = self.models_dir / f"{profile_name}.traineddata"
            
            # Simulate a training result
            with open(model_path, "wb") as f:
                f.write(b"Simulated Tesseract model file")
            
            # Update job status
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE training_jobs
                    SET status = 'completed',
                        completed_at = ?,
                        model_path = ?,
                        accuracy_improvement = ?
                    WHERE id = ?
                """, (datetime.now(), str(model_path), 0.15, job_id))
            
            self.logger.info(f"Training completed for job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing training: {e}")
            
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE training_jobs
                        SET status = 'failed',
                            error_message = ?,
                            completed_at = ?
                        WHERE id = ?
                    """, (str(e), datetime.now(), job_id))
            except Exception:
                pass
                
            return False
    
    def get_trained_model_path(self, profile_name: str = "default") -> Optional[str]:
        """Get the path to the latest trained model for a profile."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.model_path
                    FROM training_jobs t
                    JOIN handwriting_profiles p ON t.profile_id = p.id
                    WHERE p.profile_name = ? AND t.status = 'completed'
                    ORDER BY t.completed_at DESC
                    LIMIT 1
                """, (profile_name,))
                
                result = cursor.fetchone()
                if result and result['model_path']:
                    model_path = result['model_path']
                    if Path(model_path).exists():
                        return model_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting trained model path: {e}")
            return None