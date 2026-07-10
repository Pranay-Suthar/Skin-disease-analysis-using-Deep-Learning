import requests, math, json

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def fetch_nearby_hospitals(lat, lon):
    headers = {"User-Agent": "SkinDiseaseChecker/1.0"}
    found = []
    seen = set()
    for delta in [0.2, 0.5, 1.0, 2.0]:
        viewbox = f"{lon-delta},{lat+delta},{lon+delta},{lat-delta}"
        for amenity in ["hospital", "clinic"]:
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": amenity, "format": "json", "limit": 10,
                            "viewbox": viewbox, "bounded": 1, "addressdetails": 1},
                    headers=headers, timeout=10)
                for item in resp.json():
                    name = item.get("display_name", "").split(",")[0].strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    h_lat, h_lon = float(item["lat"]), float(item["lon"])
                    found.append({"name": name, "lat": h_lat, "lon": h_lon,
                                  "distance_km": round(haversine_km(lat, lon, h_lat, h_lon), 2)})
            except Exception as e:
                print(f"  ERROR: {e}")
        found.sort(key=lambda x: x["distance_km"])
        if len(found) >= 3:
            break
    return found[:3]

for city, lat, lon in [("Ahmedabad", 23.0215374, 72.5800568),
                        ("New York", 40.7128, -74.0060),
                        ("Gandhinagar", 23.2156, 72.6369)]:
    print(f"\n{city}:")
    results = fetch_nearby_hospitals(lat, lon)
    for h in results:
        print(f"  {h['name']} — {h['distance_km']} km")

with open("test_results.json", "w") as f:
    json.dump({"done": True}, f)
print("\nDone!")
