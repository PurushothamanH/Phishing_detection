import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path =model_path)
            preprocessor = load_object(file_path =preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        NumDots:int,
        PathLevel:int,
        UrlLength:int,
        NumDash:int,
        NumNumericChars:int,
        RandomString:int,
        DomainInPaths:int,
        PathLength:int,
        PctExtHyperlinks:int,
        InsecureForms:int,
        RelativeFormAction:int,
        PctNullSelfRedirectHyperlinks:int,
        FrequentDomainNameMismatch:int,
        SubmitInfoToEmail:int,
        IframeOrFrame:int,
        UrlLengthRT:int,
        PctExtResourceUrlsRT:int,
        ExtMetaScriptLinkRT:int,
        PctExtNullSelfRedirectHyperlinksRT:int
    ):
        self.NumDots = NumDots
        self.PathLevel = PathLevel
        self.UrlLength = UrlLength
        self.NumDash = NumDash
        self.NumNumericChars = NumNumericChars
        self.RandomString = RandomString
        self.DomainInPaths = DomainInPaths
        self.PathLength = PathLength
        self.PctExtHyperlinks = PctExtHyperlinks
        self.InsecureForms = InsecureForms
        self.RelativeFormAction = RelativeFormAction
        self.PctNullSelfRedirectHyperlinks = PctNullSelfRedirectHyperlinks
        self.FrequentDomainNameMismatch = FrequentDomainNameMismatch
        self.SubmitInfoToEmail = SubmitInfoToEmail
        self.IframeOrFrame = IframeOrFrame
        self.UrlLengthRT = UrlLengthRT
        self.PctExtResourceUrlsRT = PctExtResourceUrlsRT
        self.ExtMetaScriptLinkRT = ExtMetaScriptLinkRT
        self.PctExtNullSelfRedirectHyperlinksRT = PctExtNullSelfRedirectHyperlinksRT

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "NumDots":[self.NumDots],
                "PathLevel":[self.PathLevel],
                "UrlLength":[self.UrlLength],
                "NumDash":[self.NumDash],
                "NumNumericChars":[self.NumNumericChars],
                "RandomString":[self.RandomString],
                "DomainInPaths":[self.DomainInPaths],
                "PathLength":[self.PathLength],
                "PctExtHyperlinks":[self.PctExtHyperlinks],
                "InsecureForms":[self.InsecureForms],
                "RelativeFormAction":[self.RelativeFormAction],
                "PctNullSelfRedirectHyperlinks":[self.PctNullSelfRedirectHyperlinks],
                "FrequentDomainNameMismatch":[self.FrequentDomainNameMismatch],
                "SubmitInfoToEmail":[self.SubmitInfoToEmail],
                "IframeOrFrame":[self.IframeOrFrame],
                "UrlLengthRT":[self.UrlLengthRT],
                "PctExtResourceUrlsRT":[self.PctExtResourceUrlsRT],
                "ExtMetaScriptLinkRT":[self.ExtMetaScriptLinkRT],
                "PctExtNullSelfRedirectHyperlinksRT":[self.PctExtNullSelfRedirectHyperlinksRT]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

