# db_migration_tool.py
"""Command-line tool for database migrations."""

import argparse
import sys
from schema_manager import SchemaManager

def main():
    parser = argparse.ArgumentParser(description="Database Migration Tool")
    
    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check database status')
    status_parser.add_argument('--db-path', default='/mnt/rectangularfile/pdf_index.db', 
                              help='Path to the database file')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Apply database migrations')
    migrate_parser.add_argument('--db-path', default='/mnt/rectangularfile/pdf_index.db', 
                               help='Path to the database file')
    migrate_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be done without making changes')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize a new database')
    init_parser.add_argument('--db-path', default='/mnt/rectangularfile/pdf_index.db', 
                            help='Path to the database file')
    init_parser.add_argument('--dry-run', action='store_true',
                            help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Create schema manager
    schema_manager = SchemaManager(args.db_path)
    
    if args.command == 'status':
        # Show database status
        import sqlite3
        import os
        
        if not os.path.exists(args.db_path):
            print(f"Database file not found: {args.db_path}")
            return 1
        
        try:
            conn = sqlite3.connect(args.db_path)
            current_version = schema_manager.get_current_db_version(conn)
            
            # Get available migrations
            migrations = schema_manager.get_migrations()
            latest_version = max(m["version"] for m in migrations) if migrations else 0
            
            # Check what migrations are pending
            pending = [m for m in migrations if m["version"] > current_version]
            
            print(f"Database: {args.db_path}")
            print(f"Current version: {current_version}")
            print(f"Latest available version: {latest_version}")
            print(f"Pending migrations: {len(pending)}")
            
            if pending:
                print("\nPending migrations:")
                for m in pending:
                    print(f"  Version {m['version']}: {m['description']}")
            
            conn.close()
        except Exception as e:
            print(f"Error checking database status: {e}")
            return 1
    
    elif args.command == 'migrate':
        # Apply migrations
        import sqlite3
        import os
        
        if not os.path.exists(args.db_path):
            print(f"Database file not found: {args.db_path}")
            return 1
            
        try:
            conn = sqlite3.connect(args.db_path)
            current_version = schema_manager.get_current_db_version(conn)
            
            # Get available migrations
            migrations = schema_manager.get_migrations()
            
            # Check what migrations are pending
            pending = [m for m in migrations if m["version"] > current_version]
            
            if not pending:
                print("Database is already at the latest version. No migrations to apply.")
                conn.close()
                return 0
                
            print(f"Found {len(pending)} pending migrations:")
            for m in pending:
                print(f"  Version {m['version']}: {m['description']}")
                
            if args.dry_run:
                print("\nDRY RUN: No changes have been made to the database.")
                conn.close()
                return 0
                
            print("\nApplying migrations...")
            
            success = schema_manager.check_and_apply_migrations()
            if success:
                print("Migrations applied successfully!")
            else:
                print("Failed to apply migrations.")
                return 1
                
            conn.close()
            
        except Exception as e:
            print(f"Error during migration: {e}")
            return 1
    
    elif args.command == 'init':
        # Initialize database
        import os
        
        if os.path.exists(args.db_path):
            print(f"Database already exists: {args.db_path}")
            
            if args.dry_run:
                print("\nDRY RUN: This command would prompt to delete and recreate the database.")
                return 0
                
            response = input("Do you want to delete and recreate it? (y/N): ")
            if response.lower() != 'y':
                print("Aborting.")
                return 0
                
            try:
                os.remove(args.db_path)
                print(f"Deleted existing database: {args.db_path}")
            except Exception as e:
                print(f"Error deleting database: {e}")
                return 1
        else:
            print(f"Database file does not exist: {args.db_path}")
            if args.dry_run:
                print("\nDRY RUN: This command would create a new database with the following tables:")
                for table_name in schema_manager.get_base_schema().keys():
                    print(f"  - {table_name}")
                return 0
        
        if args.dry_run:
            print("\nDRY RUN: No changes have been made to the database.")
            return 0
            
        success = schema_manager.initialize_database()
        if success:
            print(f"Database initialized successfully: {args.db_path}")
        else:
            print("Failed to initialize database.")
            return 1
    else:
        parser.print_help()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())