import pymongo
class Database:
    def __init__(self, link="mongodb://localhost:27017/"):
        #Put your stuff inside a key.txt
        print("Connecting to",link)
        self.client = pymongo.MongoClient(link)
        #Creates if doesn't exist
        self.name = "test"
        self.database = self.client[self.name]
        self.accounts = self.database["accounts"]

###Account Management System#########################################
    def addAccount(self, name, password):
        #print("addAccount",name,password)
        if self.hasAccount(name):
            return False
        self.accounts.insert_one({
            "name":name, "password":guard.hash_password(password)
        })
        return True

    def hasAccount(self, name):
        myquery = { "name": name }
        user = self.accounts.find(myquery)
        #self.debug()
        return user.count()

    def getUser(self, name):
        myquery = { "name": name }
        getuser = self.accounts.find_one(myquery)
        return getuser

db = Database()

def register(name, password):
    return db.addAccount(name, password)

# A generic user model that might be used by an app powered by flask-praetorian
guard = None
class User():
    username = ""
    hashed_password = ''#guard.hash_password("1234")
    roles = ""
    is_active = True #False


    ### Individual User ##################$######
    def __init__(self, 
                 username="", hashed_password=None):
        self.username = username
        if hashed_password == None:
            self.hashed_password = guard.hash_password("1234")
        else:
            self.hashed_password = hashed_password
        self.is_active = True
        #print(self.hashed_password)

    @property
    def identity(self):
        """
        *Required Attribute or Property*
        flask-praetorian requires that the user class has an ``identity`` instance
        attribute or property that provides the unique id of the user instance
        """
        return self.username

    @property
    def rolenames(self):
        """
        *Required Attribute or Property*
        flask-praetorian requires that the user class has a ``rolenames`` instance
        attribute or property that provides a list of strings that describe the roles
        attached to the user instance
        """
        try:
            return self.roles.split(",")
        except Exception:
            return []

    @property
    def password(self):
        """
        *Required Attribute or Property*
        flask-praetorian requires that the user class has a ``password`` instance
        attribute or property that provides the hashed password assigned to the user
        instance
        """
        return self.hashed_password


    ### Multiple Users $###############################
    @classmethod
    def lookup(cls, username):
        """
        *Required Method*
        flask-praetorian requires that the user class implements a ``lookup()``
        class method that takes a single ``username`` argument and returns a user
        instance if there is one that matches or ``None`` if there is not.
        """
        userdata = db.getUser(username)
        return cls(username,userdata['password'])

    @classmethod
    def identify(cls, id):
        """
        *Required Method*
        flask-praetorian requires that the user class implements an ``identify()``
        class method that takes a single ``id`` argument and returns user instance if
        there is one that matches or ``None`` if there is not.
        """
        username = id
        userdata = db.getUser(username)
        return cls(username,userdata['password'])

    def is_valid(self):
        return self.is_active

