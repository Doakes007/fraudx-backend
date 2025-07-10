def preprocess_input(data):
    return [
        data["step"],
        data["amount"],
        data["oldbalanceOrg"],
        data["newbalanceOrig"],
        data["oldbalanceDest"],
        data["newbalanceDest"],
        1 if data["type"] == "TRANSFER" else 0,
        1 if data["type"] == "CASH_OUT" else 0
    ]

