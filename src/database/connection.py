
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Cargar variables del archivo .env
load_dotenv()


class MySQLConnector:
    def __init__(self):
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", "localhost")
        port = int(os.getenv("DB_PORT", "3306"))
        db_name = os.getenv("DB_NAME", "")

        if not db_name:
            raise ValueError("DB_NAME no está configurado en el archivo .env")

        # Construcción segura de la URL de conexión
        connection_url = URL.create(
            drivername="mysql+pymysql",
            username=user,
            password=password,
            host=host,
            port=port,
            database=db_name
        )

        self.engine = create_engine(
            connection_url,
            pool_pre_ping=True
        )

    def extract_data(self, query: str) -> pd.DataFrame:
        """Ejecuta una consulta SQL y retorna un DataFrame."""
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(text(query), connection)

                print(
                    f"[OK] Extracción exitosa. "
                    f"Registros obtenidos: {len(df)}"
                )

                return df

        except Exception as e:
            print(f"[ERROR] Error al consultar MySQL: {e}")
            raise

    def close(self):
        """Cierra las conexiones del motor."""
        self.engine.dispose()
