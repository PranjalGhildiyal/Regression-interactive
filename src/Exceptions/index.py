class NoDataPresentException(Exception):
    def __init__(self, message = 'No data present.'):
        self.message = message
        super().__init__(self.message)

