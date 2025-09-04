from app import app, db

with app.app_context():
    # Print all registered models and their corresponding tables
    print("Registered models and tables:")
    for table_name, table in db.metadata.tables.items():
        print(f"Table: {table_name}, Model: {table}")

    # Create all tables
    db.create_all()
    print("Tables created successfully!")