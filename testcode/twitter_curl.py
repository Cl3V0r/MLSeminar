import pycurl #conda install pycurl
import json
import io  

STREAM_URL = "http://chirpstream.twitter.com/2b/user.json"
REST_URL = "http://api.twitter.com/1/"


class Client:
  def __init__(self):
    self.friends = []
    self.buffer = ""
    self.userid = None
    self.conn = pycurl.Curl()

  def authenticate(self, username, password):
    output = io.StringIO()
    self.conn.setopt(pycurl.USERPWD, "%s:%s" % (username, password))
    self.conn.setopt(pycurl.URL, REST_URL + "account/verify_credentials.json")
    self.conn.setopt(pycurl.WRITEFUNCTION, output.write)
    self.conn.perform()

    data = json.loads(output.getvalue())
    if "error" in data:
      return False
    self.userid = data["id"]
    return True

  def connect(self):
    self.conn.setopt(pycurl.URL, STREAM_URL)
    self.conn.setopt(pycurl.WRITEFUNCTION, self.on_receive)
    self.conn.perform()

  def on_receive(self, data):
    self.buffer += data
    if data.endswith("rn") and self.buffer.strip():
      content = json.loads(self.buffer)
      self.buffer = ""
  
      if "friends" in content:
        self.friends = content["friends"]
  
      elif "event" in content and content["event"] == "follow":
        if content["target"]["id"] == self.userid:
          print("New follower:", content["source"]["id"])
        elif content["source"]["id"] == self.userid:
          self.friends.append(content["target"]["id"])
  
      elif "text" in content:
        to = content["in_reply_to_user_id"]
        if to and to != self.userid and to not in self.friends:
          return
        if to == self.userid:
          print("(REPLY)"),
        print(u"{0[user][name]}: {0[text]}".format(content))


client = Client()
if client.authenticate("segphault", "XXXXXXXXX"):
  client.connect()
else:
  print("Login credentials aren't valid!")
