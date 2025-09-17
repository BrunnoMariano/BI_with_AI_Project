# db_utils.py
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError

def introspect_schema(uri: str, max_tables: int = 100):
    """
    Conecta ao banco via SQLAlchemy e retorna um texto resumido do esquema
    e um dicionário {table: [(col, type), ...], ...}
    """
    engine = create_engine(uri)
    inspector = inspect(engine)
    schema_dict = {}
    schema_lines = []
    try:
        try:
            tables = inspector.get_table_names()
        except Exception:
            # Em alguns casos, usar get_table_names(schema='public') pode ser necessário
            tables = inspector.get_table_names(schema='public')

        # limita para evitar prompts enormes
        tables_to_read = tables[:max_tables]

        for t in tables_to_read:
            try:
                cols = inspector.get_columns(t)
                col_pairs = []
                for c in cols:
                    col_name = c.get("name")
                    col_type = str(c.get("type"))
                    col_pairs.append((col_name, col_type))
                schema_dict[t] = col_pairs
                cols_str = ", ".join([f"{n} ({typ})" for n, typ in col_pairs])
                schema_lines.append(f"{t}: {cols_str}")
            except Exception:
                # se não foi possível obter colunas da tabela, apenas registre o nome
                schema_dict[t] = []
                schema_lines.append(f"{t}: (colunas não disponíveis)")

        if len(tables) > max_tables:
            schema_lines.append(f"... e mais {len(tables) - max_tables} tabelas omitidas no resumo.")

        schema_text = "\n".join(schema_lines) if schema_lines else "(nenhuma tabela encontrada)"
        return schema_text, schema_dict

    except SQLAlchemyError as e:
        raise RuntimeError(f"Erro SQLAlchemy durante introspecção: {e}")
    finally:
        try:
            engine.dispose()
        except Exception:
            pass
