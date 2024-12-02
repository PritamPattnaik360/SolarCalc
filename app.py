from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# Serve the index.html file
@app.route('/')
def index():
    return render_template('index2.html')  

# Handle the calculation logic
@app.route('/calculate', methods=['POST'])
def calculate():
    # Get the request data (from the frontend form submission)
    data = request.get_json()

    # Extract data from the request
    monthly_consumption = data['monthlyConsumption']
    system_size = data['systemSize']
    installation_cost = data['installationCost']
    sunlight_hours = data['sunlightHours']
    battery_cost = data.get('batteryCost', 0)
    battery_capacity = data.get('batteryCapacity', 0)
    financing_option = data.get('financingOption', 'none')

    # Constants
    average_electricity_cost_nj = 0.14  # $/kWh (New Jersey's average electricity cost)

    # Calculation logic
    energy_savings = monthly_consumption * 12 - (system_size * sunlight_hours * 365)  # kWh per year
    annual_savings = energy_savings * average_electricity_cost_nj  # Annual savings in $
    payback_period = installation_cost / annual_savings  # Payback period in years

    # Optional: Calculate battery savings (simplified logic)
    battery_savings = battery_capacity * 365 * average_electricity_cost_nj  # Savings from battery storage in $

    # Example: Tax incentives (e.g., a fixed incentive amount)
    total_incentives = 2000  # Example fixed tax incentives in $

    # Net installation cost after applying incentives
    net_installation_cost = installation_cost - total_incentives

    # If financing is selected, calculate monthly loan payment
    loan_monthly_payment = (annual_savings / 12) if financing_option == 'loan' else 0

    # Return the calculated values as a JSON response
    return jsonify({
        'energySavings': round(energy_savings, 2),
        'annualSavings': round(annual_savings, 2),
        'totalSavingsWithBattery': round(annual_savings + battery_savings, 2),
        'paybackPeriod': round(payback_period, 2),
        'batterySavings': round(battery_savings, 2),
        'netInstallationCost': round(net_installation_cost, 2),
        'loanMonthlyPayment': round(loan_monthly_payment, 2),
        'totalIncentives': round(total_incentives, 2)
    })



if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run the app on port 5000
