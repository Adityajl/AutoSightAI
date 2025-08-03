def assess_risk(object_label, distance):
    if distance is None:
        return "Unknown"

    if object_label.lower() in ["person", "pedestrian"]:
        if distance < 2:
            return "High"
        elif distance < 5:
            return "Medium"
        else:
            return "Low"

    elif object_label.lower() in ["car", "truck", "bus"]:
        if distance < 2:
            return "Medium"
        elif distance < 5:
            return "Low"
        else:
            return "Ignore"

    elif object_label.lower() in ["bicycle", "motorcycle"]:
        if distance < 2:
            return "High"
        elif distance < 5:
            return "Medium"
        else:
            return "Low"

    else:
        return "Ignore"
