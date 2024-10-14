from pydantic import BaseModel

class Cameras(BaseModel):
    rtsp_cam : str
    isActivated : bool
    date_added : str
    location : str
    idCustomer : str