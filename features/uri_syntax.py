from re import sub

class UrlAnalyisis:
    def __init__(self, raw_url: str):
        """
        initialize the url analysis object and calculate the stadistics
        
        args:
            raw_url: → receive a raw url

        output:
            None

        time complexity → o(n)
        """
        self.url_length: int = len(raw_url) # calculate the length of the url
        self.url_path_splited: list[str] = self._split_url_path(raw_url)
        self.path_depth: int = len(self.url_path_splited)
        
        # check if the path list is not empty
        if self.url_path_splited:
            last_path: str =  self.url_path_splited[-1] # save the last path element
            # find if the path have a query 
            if '?' in last_path:
                # iterate for each last path element
                for idx, char in enumerate(last_path):
                    # check if the current index is a query
                    if char == '?':
                        self.query: str | list[float] = last_path[idx:]
                        self.url_path_splited[-1] = last_path[:idx]
                        break

            else:
                self.query: str = None

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
    """run this block when the code is runed directly"""

    # create a deept analysis of the url
    url_analysis = UrlAnalyisis('https://www.ibm.com/cloud/learn/what-is-hybrid-cloud/user?=id1245')

    # structure the object atributes
    url_analysis_dict = {
        'url_length': url_analysis.url_length,
        'url_path_splited': url_analysis.url_path_splited,
        'path_depth': url_analysis.path_depth,
        'query': url_analysis.query
    }

    print(url_analysis_dict)