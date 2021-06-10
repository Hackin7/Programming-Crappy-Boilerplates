from flask_login import UserMixin
import random

# silly user model
class User(UserMixin):
    def __init__(self, name, password):
        self.id = name
        self.name = name
        self.password = password
        
    def __repr__(self):
        return "%s/%s/%s" % (self.id, self.name, self.password)

class UserManager:
	def __init__(self):
		#self.
		pass
		
	def get(self, name, password):
		# Find user with same name
		return User(name, password)
		
	def get_session(self, name):
		# Find user with same name
		return User(name, "")
		
	def register(self, name, password):
		if self.get(name, password):
			raise Exception("Username already registered")
		else:
			pass	
		
	def delete(self, name, password):
		pass
