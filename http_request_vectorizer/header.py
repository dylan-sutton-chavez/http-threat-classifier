from re import search, sub

class Header:
    def __init__(self, user_agent: str, referer: str, host: str):
        lower_user_agent: str = user_agent.lower()

        self.user_agent: str | list[float] = user_agent
        self.browser_type: float = self._browser_type(lower_user_agent)
        self.os_name: float = self._os_name(lower_user_agent)

        self.referer_splited: list[str] | list[float] = self._split_url_path(referer)
        self.referer: str | list[float] = referer
        self.referer_length: int = len(self.referer)
        self.referer_depth: int = len(self.referer_splited)

        self.host: str | list[float] = host
        self.host_length: int = len(host)

        self.content_type: str = None
        self.cookie: str = None
        self.authorization: str = None

    def _os_name(self, lower_user_agent: str):
        """
        encode te current operative system

        args:
            lower_user_agent: str → receive the user agent in lowers

        output:
            float → hot encoding of the opeartive system

        time complexity → o(n)
        """
        normal_os: str = 'windows|macintosh|linux|iphone|cros|ipad'

        hot_encoding_bag: dict[str, int] = {
            "windows": 1.0,
            "macintosh": 2.0,
            "linux": 3.0,
            "iphone": 4.0,
            "ipad": 5.0,
            "cros": 6.0
        }

        os = search(normal_os, lower_user_agent)
        return hot_encoding_bag[os.group(0)] if os else 0.0

    def _browser_type(self, lower_user_agent: str):
        """
        search the browser type in the user agent

        args:
            lower_user_agent: str → receive the user agent in lowers

        output:
            float → return 1 if its a normal browser, if its not return 0

        time complexity → o(n)
        """
        normal_browsers: str = 'chrome|firefox|edg|edge|safari|opr|vivaldi|samsungbrowser|msie|trident|mobile|torbrowser|brave'
        return 1.0 if search(normal_browsers, lower_user_agent) else 0.0
    
    def _split_url_path(self, raw_url: str):
        """
        split a raw url into a paths list
        
        args:
            raw_url: → receive a raw url

        output:
            list[str] → return a list with the paths

        time complexity → o(n)
        """
        path_only = sub(r'^(?:\w+://)?[^/]+', '', raw_url)
        stripped_path = path_only.strip('/')
        
        return stripped_path.split('/')
    
if __name__ == '__main__':
    header = Header('Mozilla/5.0 (Windows NT 10.0; Win64; x64; 192.168.0.15) Chrome/122.0.6261.95 Safari/537.36')
    print(header.browser_type, header.os_name)
