import logging
import pyodbc

class User:

	def __init__(self,userId):
		self.UserId = userId
		self.IsInDb = False
		self.StatusId = 1

	def Check(self,connectionStringUsers):

		SqlInput = f"SELECT COUNT(*) AS [IsUserInDB], MAX([StatusId]) AS [StatusId] FROM [Users].[Values] WHERE [Guid] = '{self.UserId}'"

		try:
			cnxn = pyodbc.connect(connectionStringUsers)
			cursor = cnxn.execute(SqlInput)
			for row in cursor.fetchall():
				self.IsInDb = row[0]
				self.StatusId = row[1]
			cnxn.close()
		except Exception as e:
			logging.error(e)
			logging.info(SqlInput)