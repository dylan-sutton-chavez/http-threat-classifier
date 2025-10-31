class ClientProfile:
    def __init__(self, method: str):
        self.origin_ip: str = None
        self.geolocalization: str = None
        self.http_method: float = self._method_encoder(method)
        self.request_rate: int = None
        self.errors_rate: int = None
        self.device_type: int = None
        self.browser: int = None
        self.port: int = None

    def _method_encoder(method: str):
        """
        encode the method in a floatant value

        args:
            method: str → method of the user

        output:
            float → encoded value of the method

        time complexity → o(n)
        """
        method_lower: str = method.lower()

        # encoded actions bag
        actions_bag = {
            "get": 1.0,
            "head": 1.0,
            "post": 1.0,
            "put": 0.5,
            "patch": 0.5,
            "delete": 0.5,
            "options": 0.5,
        }

        # confirm if the method is in the actions bag
        if method_lower in actions_bag:
            return actions_bag[method_lower]
        
        return 0.0