import psycopg2
import psql_connector 


class Databases():
    def __init__(self):
        self.db = psycopg2.connect(host='localhost', dbname='test_db', user='soobinlee',password='123456',port=5432)
        self.cursor = self.db.cursor()

    def __del__(self):
        self.db.close()
        self.cursor.close()

    def execute(self,query,args={}):
        self.cursor.execute(query,args)
        row = self.cursor.fetchall()
        return row

    def commit(self):
        self.cursor.commit()


class CRUD(Databases):
    def insert_db(self,table,colum,data, schema="public"):
        sql = " INSERT INTO {schema}.{table}({colum}) VALUES ('{data}') ;".format(schema=schema,table=table,colum=colum,data=data)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e :
            print(" insert DB err ",e) 
    
    def read_db(self,table, colum, schema="public"):
        sql = " SELECT {colum} from {schema}.{table}".format(colum=colum,schema=schema,table=table)
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
        except Exception as e :
            result = (" read DB err",e)
        
        return result

    def update_db(self,table,colum,value,condition, schema="public"):
        sql = " UPDATE {schema}.{table} SET {colum}='{value}' WHERE {colum}='{condition}' ".format(schema=schema
        , table=table , colum=colum ,value=value,condition=condition )
        try :
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e :
            print(" update DB err",e)

    def delete_db(self,table,condition, schema="public"):
        sql = " delete from {schema}.{table} where {condition} ; ".format(schema=schema,table=table,
        condition=condition)
        try :
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print( "delete DB err", e)


db_connector = CRUD()
print(db_connector.read_db(table="test_table", colum="*"))