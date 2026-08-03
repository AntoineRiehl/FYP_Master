#src/atlas/schema/feature_config.py

from dataclasses import dataclass


@dataclass
class FeatureConfig:


    name: str



    # Text features

    use_tags: bool = True

    use_categories: bool = True

    use_reviews: bool = False



    # Metadata features

    use_metadata: bool = True

    use_statistics: bool = True



    # Future

    use_images: bool = False

    use_external_embeddings: bool = False



    def description(self):

        features=[]


        if self.use_tags:
            features.append("tags")

        if self.use_categories:
            features.append("categories")

        if self.use_reviews:
            features.append("reviews")

        if self.use_metadata:
            features.append("metadata")


        return "+".join(features)