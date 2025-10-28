from re import sub

class UrlAnalyisis:
    def __init__(self, raw_url: str):
        self.url_length: int = len(raw_url) # calculate the length of the url

        self.url_path_splited: list[str] = self._split_url_path(raw_url)

        self.path_depth: int = len(self.url_path_splited)
        

        last_path: str =  self.url_path_splited[-1]

        if '?' in last_path:
            for idx, char in enumerate(last_path):
                if char == '?':
                    self.query: str | list[float] = last_path[idx:]
                    break
                    
        else:
            self.query: str = None

    def _split_url_path(self, raw_url: str):
        path_only = sub(r'^(?:\w+://)?[^/]+', '', raw_url)
        stripped_path = path_only.strip('/')
        
        return stripped_path.split('/')

if __name__ == '__main__':
    """run this block when the code is runed directly"""

    url_analysis = UrlAnalyisis('https://www.ibm.com/cloud/learn/what-is-hybrid-cloud')

    print(url_analysis.path_depth)
