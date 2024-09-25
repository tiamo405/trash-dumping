ERROR_RESPS = {
    0: ["FS.0000", "Success."],
    1: ["FS.0001", "Unknown error."],

    # add camera
    100: ["FS.0100", "Success add camera"],
    101: ["FS.0101", "Camra already exists."],
    102: ["FS.0102", "Camera don't recognize."],

    # remove camera
    200: ["FS.0200", "Success remove camera"],
    201: ["FS.0201", "Camera does not exist."],

    # docker
    300: ["FS.0300", "Success remove container"],
    301: ["FS.0301", "Error remove container"],

    # history
    400: ["FS.0400", "Success get history"],
    401: ["FS.0401", "Error get history"],
    402: ["FS.0402", "Camera does not exist."],
      
}


class Response:
    def __init__(self, code, msg="", error_resp=-1, entities={}):
        if error_resp > 0:
            self.msg = ERROR_RESPS[error_resp][1]
            self.error_code = ERROR_RESPS[error_resp][0]
        else:
            self.msg = ""
            self.error_code = None

        self.code = code
        self.msg = msg if len(msg) > 0 else self.msg
        self.entities = entities
    
    def update_error_resp(self, error_resp):
        self.msg = ERROR_RESPS[error_resp][1]
        self.error_code = ERROR_RESPS[error_resp][0]