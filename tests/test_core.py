"""Tests for Arogya."""
from src.core import Arogya
def test_init(): assert Arogya().get_stats()["ops"] == 0
def test_op(): c = Arogya(); c.detect(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Arogya(); [c.detect() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Arogya(); c.detect(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Arogya(); r = c.detect(); assert r["service"] == "arogya"
