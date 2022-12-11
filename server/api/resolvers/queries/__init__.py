from ariadne import QueryType
from api.types import * 


query = QueryType()
@query.field("meta")
def meta_resolver(obj, info):
   return Meta(
        programmer = "@crispengari",
        main = "Wild Animal Classification (WAC)",
        description = "given an image of a wild animal, the API should be able to predict the name of an animal among the 6.",
        language = "python",
        libraries = ["pytorch", "torchvision"],
   ).to_json()
   