import time
import datetime

class Convtime:
    def convert_time_to_unix(dt):
        """
        Converts time to unix
        
        Parameters:
        ----------
        dt: string
            datetime in %Y-%m-%d %H:%M:%S format
        """
        dt = dt.split('.')

        dt = datetime.datetime.strptime(dt[0],"%Y-%m-%d %H:%M:%S")
        return time.mktime(dt.timetuple())

