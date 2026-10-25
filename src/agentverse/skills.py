"""Explicit skill registry; no transfer between independently trained tables."""

from .world import CONTRACT as NAVIGATION_CONTRACT, NavigationWorld
from .collection import CONTRACT as COLLECTION_CONTRACT, ResourceCollectionWorld

SKILLS = {"navigation": (NavigationWorld, NAVIGATION_CONTRACT),
          "resource_collection": (ResourceCollectionWorld, COLLECTION_CONTRACT)}


def skill_spec(skill):
    if skill not in SKILLS:
        raise ValueError(f"Unknown skill: {skill}")
    return SKILLS[skill]
