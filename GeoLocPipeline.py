import argparse
import gc
from rag_retriver import retrieve_similar_images
from sentence_transformers import SentenceTransformer

from infer_finetuned import InferencePipeline
from infer import infer_model 


class GeoLocReasonPipeline:
     def __init__(self, rag_index_path, 
                  rag_images_path, 
                  rag_model, 
                  reasoner_model_path = "./Qwen-VL/Qwen-VL-Models/Qwen-VL-Chat",
                  inference_model_path = "./Qwen-VL/Qwen-VL-Models/Qwen2-VL-Chat-Finetuned" , 
                  inference_adapter_path = "./Qwen-VL/Qwen-VL-Adapters/qwen2-vl-chat-finetuned-geolocrag-adapter"):
         self.rag_model = rag_model
         self.rag_index_path = rag_index_path
         self.rag_images_path = rag_images_path
         self.reasoner_model_path = reasoner_model_path
         self.inference_model_path = inference_model_path
         self.inference_adapter_path = inference_adapter_path
         self.inference_pipe = InferencePipeline(base_model_path = self.inference_model_path, adapter_path =self.inference_adapter_path)

     def call_resoner_model(self,query_image):
         """
         Call the a reasoner model to get desciption of the place
         We can call the GeoReasoner Pipeline from 
         """
         response, history = infer_model(query_image,self.reasoner_model_path)
         return response
     def call_rag_model(self, query):
         """
         Use the RAG pipline to get some candidate images for the reasoner model
         """
         # for this pipeline assume only returnig the top 1-result
         images, distance =  retrieve_similar_images(query,self.rag_model, self.rag_index_path,top_k = 3)
         return images[0]
     def call_other_api(self):
         """
         call wiki, openstreetview, google street view to get results
         """
         pass
     def final_predictor_model(self, reason, query_image):
         
        instruction = (
            "You are expert in analyzing image and predicting the location from the scence identify common clues to identify the locaition and "
            "return the contient, country, city , latitude longtiude of the given image output a json format as follow don't output extra result." \
            f"Given the reasoning here{reason}"
            "{result:{\"country\":\"\", \"city\":\"\", \"latitude\":, \"longitude\":}}"
        )
        final_prediction = self.inference_pipe.predict(query_image, instruction)
        return final_prediction
     def final_predication_based_on_result(self, query_image):
         """
         concatinate the result form the above methods and pass VLM for to generate a final 
         """
         #call reasoner to get reason from the model
         #reasoning = self.call_resoner_model(query_image)
         # clean GPU to load the other models
         #gc.collect()
         #get closer candidate image
         reasoning = "Given an image craft a brief and choesive resoning path that deduces this location based on the visual clues precent in the image"
         rag_result_image = self.call_rag_model(query_image)
         #clean rag model from gpu
         gc.collect()
         # call multi modal final predictor to get final predction
         return self.final_predictor_model(reasoning, query_image)

         
         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument for the pipeline")
    parser.add_argument("--rag_index_path")
    parser.add_argument("--rag_base", type=str, required=False)
    parser.add_argument("--query_image", type=str, required=False)
    BASE_PATH_RAG = "/nobackup/riyad/NAVIG/data/"
   
    args = parser.parse_args("--query_image", type=str)
    model = SentenceTransformer('clip-ViT-B-32')
    rag_index_path = args.rag_index_path
    
    geoPipeline = GeoLocReasonPipeline(rag_index_path= rag_index_path,
                                       rag_images_path=BASE_PATH_RAG,
                                       rag_model=model)
    query_img_path = args.query_image