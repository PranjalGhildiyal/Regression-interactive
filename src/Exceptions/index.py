class NoDataPresentException(Exception):
    def __init__(self, message = 'No data present.'):
        self.message = message
        super().__init__(self.message)

class InvalidModelException(Exception):
    def __init__(self, message = 'Invalid model type entered. See config files to add models.'):
        self.message = message
        super().__init__(self.message)


