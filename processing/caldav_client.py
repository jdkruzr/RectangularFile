import caldav
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import uuid
from urllib.parse import urljoin
import os
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

class CalDAVTodoClient:
    """CalDAV client for managing todos in calendar servers"""
    
    def __init__(self):
        self.client = None
        self.calendar = None
        self.encryption_key = self._get_or_create_encryption_key()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for storing passwords"""
        key_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.caldav_key')
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Secure permissions
            return key
    
    def encrypt_password(self, password: str) -> str:
        """Encrypt password for storage"""
        fernet = Fernet(self.encryption_key)
        encrypted = fernet.encrypt(password.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_password(self, encrypted_password: str) -> str:
        """Decrypt password from storage"""
        fernet = Fernet(self.encryption_key)
        encrypted_bytes = base64.b64decode(encrypted_password.encode())
        return fernet.decrypt(encrypted_bytes).decode()
    
    def connect(self, url: str, username: str, password: str, calendar_name: str = 'todos') -> bool:
        """
        Connect to CalDAV server and select calendar
        
        Args:
            url: CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            calendar_name: Name of calendar to use for todos
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create CalDAV client
            self.client = caldav.DAVClient(
                url=url,
                username=username,
                password=password
            )
            
            # Get principal (user)
            principal = self.client.principal()
            
            # Get calendars
            calendars = principal.calendars()
            
            # Find or create the todos calendar
            self.calendar = None
            for cal in calendars:
                if cal.name == calendar_name:
                    self.calendar = cal
                    break
            
            if self.calendar is None:
                # Create new calendar for todos
                logger.info(f"Creating new calendar: {calendar_name}")
                self.calendar = principal.make_calendar(name=calendar_name)
            
            logger.info(f"Connected to CalDAV server, using calendar: {calendar_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to CalDAV server: {e}")
            self.client = None
            self.calendar = None
            return False
    
    def test_connection(self, url: str, username: str, password: str) -> Dict[str, any]:
        """
        Test CalDAV connection without storing settings
        
        Returns:
            Dict with success status and message
        """
        try:
            client = caldav.DAVClient(
                url=url,
                username=username,
                password=password
            )
            
            principal = client.principal()
            calendars = principal.calendars()
            
            return {
                'success': True,
                'message': f'Connection successful! Found {len(calendars)} calendars.'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}'
            }
    
    def create_todo(self, summary: str, description: str = '', due_date: Optional[datetime] = None, 
                   priority: int = 5, categories: List[str] = None) -> Optional[str]:
        """
        Create a new todo item
        
        Args:
            summary: Todo title/summary
            description: Detailed description
            due_date: Due date (optional)
            priority: Priority 1-9 (1=highest, 9=lowest, 5=medium)
            categories: List of category strings
            
        Returns:
            UID of created todo, or None if failed
        """
        if not self.calendar:
            logger.error("Not connected to CalDAV server")
            return None
        
        try:
            # Generate UID
            todo_uid = str(uuid.uuid4())
            
            # Build VTODO content
            categories_str = ""
            if categories:
                categories_str = f"CATEGORIES:{','.join(categories)}\n"
            
            due_str = ""
            if due_date:
                due_str = f"DUE:{due_date.strftime('%Y%m%dT%H%M%SZ')}\n"
            
            created_time = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            
            vtodo_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//RectangularFile//CalDAV Integration//EN
BEGIN:VTODO
UID:{todo_uid}
DTSTAMP:{created_time}
CREATED:{created_time}
SUMMARY:{summary}
DESCRIPTION:{description}
PRIORITY:{priority}
STATUS:NEEDS-ACTION
{due_str}{categories_str}END:VTODO
END:VCALENDAR"""
            
            # Create todo in calendar
            todo = self.calendar.save_todo(vtodo_content)
            
            logger.info(f"Created todo: {summary} (UID: {todo_uid})")
            return todo_uid
            
        except Exception as e:
            logger.error(f"Failed to create todo: {e}")
            return None
    
    def get_todos(self, include_completed: bool = False) -> List[Dict]:
        """
        Get all todos from the calendar
        
        Args:
            include_completed: Whether to include completed todos
            
        Returns:
            List of todo dictionaries
        """
        if not self.calendar:
            logger.error("Not connected to CalDAV server")
            return []
        
        try:
            todos = []
            
            # Get all todos from calendar
            todo_objects = self.calendar.todos(include_completed=include_completed)
            
            for todo in todo_objects:
                try:
                    # Parse VTODO data
                    vtodo = todo.icalendar_component
                    
                    # Convert datetime objects to ISO strings for JSON serialization
                    def convert_datetime(dt_obj):
                        if dt_obj is None:
                            return None
                        try:
                            if hasattr(dt_obj, 'isoformat'):
                                return dt_obj.isoformat()
                            elif hasattr(dt_obj, 'dt') and hasattr(dt_obj.dt, 'isoformat'):
                                return dt_obj.dt.isoformat()
                            else:
                                return str(dt_obj)
                        except:
                            return str(dt_obj)
                    
                    created = vtodo.get('CREATED')
                    due = vtodo.get('DUE')
                    
                    todo_dict = {
                        'uid': str(vtodo.get('UID', '')),
                        'summary': str(vtodo.get('SUMMARY', '')),
                        'description': str(vtodo.get('DESCRIPTION', '')),
                        'status': str(vtodo.get('STATUS', 'NEEDS-ACTION')),
                        'priority': int(vtodo.get('PRIORITY', 5)),
                        'created': convert_datetime(created),
                        'due': convert_datetime(due),
                        'categories': []
                    }
                    
                    # Parse categories
                    categories = vtodo.get('CATEGORIES')
                    if categories:
                        if isinstance(categories, str):
                            todo_dict['categories'] = [cat.strip() for cat in categories.split(',')]
                        else:
                            todo_dict['categories'] = [str(categories)]
                    
                    todos.append(todo_dict)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse todo: {e}")
                    continue
            
            logger.info(f"Retrieved {len(todos)} todos from CalDAV")
            return todos
            
        except Exception as e:
            logger.error(f"Failed to get todos: {e}")
            return []
    
    def update_todo_status(self, uid: str, completed: bool) -> bool:
        """
        Update todo completion status
        
        Args:
            uid: UID of todo to update
            completed: True to mark complete, False for incomplete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.calendar:
            logger.error("Not connected to CalDAV server")
            return False
        
        try:
            # Find the todo by UID
            todos = self.calendar.todos(include_completed=True)
            
            for todo in todos:
                vtodo = todo.icalendar_component
                if str(vtodo.get('UID')) == uid:
                    # Update status
                    status = 'COMPLETED' if completed else 'NEEDS-ACTION'
                    vtodo['STATUS'] = status
                    
                    if completed:
                        # Set completion date in proper iCalendar format
                        from icalendar import vDatetime
                        completed_time = datetime.now(timezone.utc)
                        vtodo['COMPLETED'] = vDatetime(completed_time)
                        vtodo['PERCENT-COMPLETE'] = 100
                    else:
                        # Remove completion date
                        if 'COMPLETED' in vtodo:
                            del vtodo['COMPLETED']
                        vtodo['PERCENT-COMPLETE'] = 0
                    
                    # Save changes
                    todo.save()
                    logger.info(f"Updated todo {uid} status to {status}")
                    return True
            
            logger.warning(f"Todo with UID {uid} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to update todo status: {e}")
            return False
    
    def delete_todo(self, uid: str) -> bool:
        """
        Delete a todo by UID
        
        Args:
            uid: UID of todo to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.calendar:
            logger.error("Not connected to CalDAV server")
            return False
        
        try:
            # Find and delete the todo
            todos = self.calendar.todos(include_completed=True)
            
            for todo in todos:
                vtodo = todo.icalendar_component
                if str(vtodo.get('UID')) == uid:
                    todo.delete()
                    logger.info(f"Deleted todo {uid}")
                    return True
            
            logger.warning(f"Todo with UID {uid} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete todo: {e}")
            return False