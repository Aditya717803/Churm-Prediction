def generate_offer(churn_probability, customer_name):
    if churn_probability > 0.9:
        offer = "Have a 50% off on next 3 month subscription."
    elif churn_probability > 0.8:
        offer = "Hey [Customer Name] 😎, ready for some good news? 🎉 Your next subscription just got 30% OFF! Yup, you read that right ."
    elif churn_probability > 0.7:
        offer = "Have a 20% off on next 3 month subscription."
    elif churn_probability > 0.6:
        offer = "Hey [Customer Name] 😎, ready for some good news? 🎉 Your next subscription just got 10% OFF! Yup, you read that right ."
    else:
        offer = "Thank you for using our service! Explore our new features."

    return f"Dear {customer_name}, {offer}"
