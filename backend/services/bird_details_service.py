
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BIRD_DETAILS_PATH = (
    PROJECT_ROOT / "backend" / "data" / "bird_details.json"
)


class BirdDetailsService:

    def __init__(self):
        self.bird_details = {}
        self.load_details()

    def load_details(self):
        if not BIRD_DETAILS_PATH.exists():
            raise FileNotFoundError(
                f"Bird details file not found: {BIRD_DETAILS_PATH}"
            )

        with open(BIRD_DETAILS_PATH, "r", encoding="utf-8") as file:
            self.bird_details = json.load(file)

        print(
            f"Loaded {len(self.bird_details)} bird detail records."
        )

    def get_bird_details(self, species_name: str):
        normalized_name = species_name.strip().upper()

        bird = self.bird_details.get(normalized_name)

        if bird is None:
            return None

        return {
            "success": True,
            "species_label": normalized_name,
            "bird_details": bird
        }


bird_details_service = BirdDetailsService()
