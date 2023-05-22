from datetime import date, datetime
import getpass

def creation():
    now = datetime.now().strftime("%H:%M:%S")
    today = date.today().strftime("%YY:%MM%DD")
    user = getpass.getuser()
    return {'user' : user, 'date': today, 'time': now }

