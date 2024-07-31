import mysql.connector

def create_database_and_tables():
    # Establish the connection
    cnx = mysql.connector.connect(user='root', password='admin', host='localhost')

    # Create a cursor object
    cursor = cnx.cursor()

    # Drop previous database to avoid conflicts
    drop_db_query = "DROP DATABASE IF EXISTS myapp;"
    cursor.execute(drop_db_query)

    # Create the database
    create_db_query = "CREATE DATABASE myapp;"
    cursor.execute(create_db_query)

    # Switch to the newly created database
    cnx.database = 'myapp'

    # Create the 'tasks' table
    create_table_query = """
    CREATE TABLE tasks (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        task_id VARCHAR(255) NOT NULL,
        status VARCHAR(10) NOT NULL,
        processing_duration FLOAT,
        datetime DATETIME DEFAULT CURRENT_TIMESTAMP,
        data TEXT 
    );
    """
    cursor.execute(create_table_query)

    # Commit the transaction
    cnx.commit()

    # Close the cursor and connection
    cursor.close()
    cnx.close()

# Call the function to create the database and table
create_database_and_tables()
