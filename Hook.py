class SaveInput:
    
    def __init__(self):
        self.output=None
        self.input = None
    
    def __call__(self,layer,input,output):
        self.output = output
        self.input = input[0]
    def clear(self):
        self.output=None
        self.input=None


