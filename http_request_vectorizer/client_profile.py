class ClientProfile:
    def __init__(self):
        self.origin_ip: str = None
        self.geolocalization: str = None
        self.http_method: str = None
        self.request_rate: int = None
        self.errors_rate: int = None
        self.device_type: int = None
        self.browser: int = None
