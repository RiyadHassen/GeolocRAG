class GeoLocReasonPipeline:
     def __init__(self):
         pass
     def call_resoner_model(self):
         """
         Call the a reasoner model to get desciption of the place
         We can call the GeoReasoner Pipeline from 
         """
         pass
     def call_rag_model(self):
         """
         Use the RAG pipline to get some candidate images for the reasoner model
         """
         pass
     def call_other_api(self):
         """
         call wiki, openstreetview, google street view to get results
         """
         pass
     def final_predication_based_on_result(self):
         """
         concatinate the result form the above methods and pass VLM for to generate a final 
         """
         pass