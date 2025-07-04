# db_migration_tool.py
"""Command-line tool for database migrations."""

import argparse
import sys
from db.schema_manager import SchemaManager

def initialize_version_table(db_path, target_version):
    """Initialize the version table at a specific version."""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the version table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='db_version'
    """)
    
    if not cursor.fetchone():
        # Create version table if it doesn't exist
        cursor.execute("""
            CREATE TABLE db_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("Created db_version table")
    
    # Check if the version already exists
    cursor.execute("SELECT version FROM db_version WHERE version = ?", (target_version,))
    if cursor.fetchone():
        print(f"Version {target_version} already exists in the version table")
    else:
        # Insert the target version
        cursor.execute("INSERT INTO db_version (version) VALUES (?)", (target_version,))
        print(f"Added version {target_version} to the version table")
    
    # Show all versions
    cursor.execute("SELECT version FROM db_version ORDER BY version")
    versions = [row[0] for row in cursor.fetchall()]
    print(f"Current versions in db_version table: {versions}")
    
    conn.commit()
    conn.close()
    print(f"Version table initialized at version {target_version}")

def main():
    parser = argparse.ArgumentParser(
        description="Database Migration Tool for RectangularFile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current database status
  python db_migration_tool.py status

  # Apply pending migrations
  python db_migration_tool.py migrate

  # Preview migrations without applying them
  python db_migration_tool.py migrate --dry-run

  # Initialize a new database
  python db_migration_tool.py init

  # Preview database initialization
  python db_migration_tool.py init --dry-run
  
  # View the current database schema
  python db_migration_tool.py schema
  
  # Initialize the version table at a specific version
  python db_migration_tool.py init-version --version 3
"""
    )
    
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
                            
    # Schema command
    schema_parser = subparsers.add_parser('schema', help='Print the current database schema')
    schema_parser.add_argument('--db-path', default='/mnt/rectangularfile/pdf_index.db', 
                              help='Path to the database file')
    schema_parser.add_argument('--table', help='Print schema for a specific table only')
    schema_parser.add_argument('--output', help='Save schema to a file instead of printing to console')

    # Initialize version command
    init_version_parser = subparsers.add_parser('init-version', 
                                               help='Initialize the version table at a specific version')
    init_version_parser.add_argument('--db-path', default='/mnt/rectangularfile/pdf_index.db', 
                                    help='Path to the database file')
    init_version_parser.add_argument('--version', type=int, required=True,
                                    help='Version number to initialize at')
    
    args = parser.parse_args()
    
    # If no command is provided, show help and exit
    if not args.command:
        parser.print_help()
        return 0
    
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
                    print(f"    SQL: {m['sql']}")
                
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
                
                print("\nWith the following indexes:")
                for index_name in schema_manager.get_indexes().keys():
                    print(f"  - {index_name}")
                    
                print("\nAnd set the database version to the latest migration:")
                migrations = schema_manager.get_migrations()
                latest_version = max(m["version"] for m in migrations) if migrations else 0
                print(f"  - Version {latest_version}")
                
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
            
    elif args.command == 'schema':
        # Print the database schema
        import sqlite3
        import os
        import textwrap
        
        if not os.path.exists(args.db_path):
            print(f"Database file not found: {args.db_path}")
            return 1
            
        try:
            conn = sqlite3.connect(args.db_path)
            cursor = conn.cursor()
            
            # Get list of tables
            if args.table:
                tables = [args.table]
                # Check if table exists
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (args.table,))
                if not cursor.fetchone():
                    print(f"Table '{args.table}' not found in database.")
                    return 1
            else:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                tables = [row[0] for row in cursor.fetchall()]
                if not tables:
                    print("No tables found in database.")
                    return 0
            
            # Prepare output
            output_lines = []
            output_lines.append(f"Database: {args.db_path}")
            output_lines.append(f"Tables: {len(tables)}")
            output_lines.append("")
            
            # Get schema for each table
            for table in tables:
                output_lines.append(f"Table: {table}")
                output_lines.append("-" * (len(table) + 7))
                
                # Get CREATE TABLE statement
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
                create_stmt = cursor.fetchone()[0]
                
                # Format the SQL statement for better readability
                formatted_sql = textwrap.indent(create_stmt, '  ')
                output_lines.append(formatted_sql)
                
                # Get table columns and types
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                output_lines.append("\nColumns:")
                for col in columns:
                    col_id, name, type_name, not_null, default_val, pk = col
                    constraints = []
                    if pk:
                        constraints.append("PRIMARY KEY")
                    if not_null:
                        constraints.append("NOT NULL")
                    if default_val is not None:
                        constraints.append(f"DEFAULT {default_val}")
                        
                    constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                    output_lines.append(f"  {name}: {type_name}{constraint_str}")
                
                # Get indexes for this table
                cursor.execute(f"SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND name NOT LIKE 'sqlite_%'", (table,))
                indexes = cursor.fetchall()
                
                if indexes:
                    output_lines.append("\nIndexes:")
                    for idx_name, idx_sql in indexes:
                        output_lines.append(f"  {idx_name}: {idx_sql}")
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                foreign_keys = cursor.fetchall()
                
                if foreign_keys:
                    output_lines.append("\nForeign Keys:")
                    for fk in foreign_keys:
                        fk_id, seq, ref_table, from_col, to_col, on_update, on_delete, match = fk
                        output_lines.append(f"  {from_col} -> {ref_table}({to_col})")
                        if on_update != "NO ACTION":
                            output_lines.append(f"    ON UPDATE: {on_update}")
                        if on_delete != "NO ACTION":
                            output_lines.append(f"    ON DELETE: {on_delete}")
                
                output_lines.append("\n")
            
            # Output the schema information
            schema_text = "\n".join(output_lines)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(schema_text)
                print(f"Schema saved to {args.output}")
            else:
                print(schema_text)
                
            conn.close()
            
        except Exception as e:
            print(f"Error getting database schema: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    elif args.command == 'init-version':
        # Initialize version table
        import os
        
        if not os.path.exists(args.db_path):
            print(f"Database file not found: {args.db_path}")
            return 1
            
        try:
            initialize_version_table(args.db_path, args.version)
            return 0
        except Exception as e:
            print(f"Error initializing version table: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())