from datetime import datetime, time, timedelta
import pytz

# Múi giờ Việt Nam (GMT+7)
tz_vietnam = pytz.timezone('Asia/Ho_Chi_Minh')

# get timestame
def get_current_timestamp():
    return int(datetime.now(tz_vietnam).timestamp())

# get timestamp date
def get_date_timestamp():
    # Lấy thời gian hiện tại
    now = datetime.now()
    
    # Điều chỉnh múi giờ GMT+7 (UTC +7 giờ)
    tz_offset = timedelta(hours=7)
    
    # Lấy thời gian hiện tại theo múi giờ GMT+7
    vietnam_time = now - tz_offset
    
    # Kết hợp ngày hiện tại với thời gian 00:00:00
    vietnam_midnight = datetime.combine(vietnam_time.date(), time(0, 0, 0))
    
    # Trả về timestamp của thời gian này
    return int(vietnam_midnight.timestamp())

# get datetime
def get_datetime():
    return datetime.now()

if '__main__' == __name__:
    print(get_date_timestamp())
    print(datetime.combine(datetime.now().date(), time(0, 0, 0)).timestamp())
    print(datetime.now())