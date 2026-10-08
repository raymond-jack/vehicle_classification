from app.vehicle_predictor import VehiclePredictor

def main():
    predictor = VehiclePredictor()
    label, confidence = predictor.predict(
        "testimg/bus_test.jpg"
    )
    print(label)
    print(confidence)

if __name__=="__main__":
    main()