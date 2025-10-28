from re import sub

class UrlAnalyisis:
    def __init__(self, raw_url: str):
        self.url_length: int = len(raw_url) # calculate the length of the url

        url_path_splited = self._split_url_path(raw_url)

        self.path_depth: int = len(url_path_splited)

    def _split_url_path(self, raw_url: str):
        path_only = sub(r'^(?:\w+://)?[^/]+', '', raw_url)
        stripped_path = path_only.strip('/')
        
        return stripped_path.split('/')

if __name__ == '__main__':
    """run this block when the code is runed directly"""

    url_analysis = UrlAnalyisis('https://www.ibm.com/cloud/learn/what-is-hybrid-cloud')

    print(url_analysis.path_depth)
