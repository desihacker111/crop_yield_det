from flask import Flask, render_template, request, flash
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

# Get the absolute path to the directory containing this script
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load the trained model pipeline with proper path handling
try:
    model_path = os.path.join(BASE_DIR, "model_pipeline.pkl")
    data_path = os.path.join(BASE_DIR, "yield_df.csv")
    
    model = pickle.load(open(model_path, "rb"))
    df = pd.read_csv(data_path)
except FileNotFoundError as e:
    print(f"Error: Required files not found - {str(e)}")
    model = None
    df = None

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Input validation
            try:
                year = int(request.form["year"])
                rain = float(request.form["rain"])
                pesticide = float(request.form["pesticide"])
                temp = float(request.form["temp"])
                area = request.form["area"].strip()
                crop = request.form["crop"].strip()

                # Basic validation
                if year < 1900 or year > 2100:
                    raise ValueError("Invalid year range")
                if rain < 0 or pesticide < 0 or temp < -50 or temp > 60:
                    raise ValueError("Invalid input values")
                if not area or not crop:
                    raise ValueError("Area and crop cannot be empty")

            except ValueError as e:
                return render_template("index.html", prediction_text=f"Error: {str(e)}")

            # Creating a DataFrame for the input values
            input_data = pd.DataFrame([{
                'Year': year,
                'average_rain_fall_mm_per_year': rain,
                'pesticides_tonnes': pesticide,
                'avg_temp': temp,
                'Area': area,
                'Item': crop
            }])

            # Making the prediction using the model
            try:
                prediction = model.predict(input_data)[0]
                prediction = round(prediction, 2)
            except Exception as e:
                return render_template("index.html", 
                    prediction_text=f"Error making prediction: {str(e)}")

            # Process historical data
            try:
                crop_data = df[df['Item'] == crop]
                if crop_data.empty:
                    return render_template("index.html", 
                        prediction_text="Error: No historical data for selected crop")
                    
                crop_data = crop_data.groupby('Year')['hg/ha_yield'].mean().reset_index()
                
                # Enhanced statistics with actual vs predicted
                years = crop_data['Year'].tolist()
                yields = crop_data['hg/ha_yield'].round(2).tolist()
                avg_yield = round(crop_data['hg/ha_yield'].mean(), 2)
                max_yield = round(crop_data['hg/ha_yield'].max(), 2)
                min_yield = round(crop_data['hg/ha_yield'].min(), 2)
                std_yield = round(crop_data['hg/ha_yield'].std(), 2)
                growth_rate = round(((yields[-1] - yields[0]) / yields[0]) * 100, 2)
                
                # Calculate 5-year trend
                recent_trend = round(((yields[-1] - yields[-5]) / yields[-5]) * 100, 2) if len(yields) >= 5 else 0
                
                # Prediction comparison
                pred_vs_avg = round(((prediction - avg_yield) / avg_yield) * 100, 2)
                pred_vs_last = round(((prediction - yields[-1]) / yields[-1]) * 100, 2)

                # Get actual yields for comparison
                actual_yields = df[(df['Item'] == crop) & (df['Area'] == area)]['hg/ha_yield'].tolist()
                actual_years = df[(df['Item'] == crop) & (df['Area'] == area)]['Year'].tolist()
                
                # Calculate error metrics if actual data exists
                if actual_yields:
                    last_actual = actual_yields[-1]
                    accuracy = round((1 - abs(prediction - last_actual) / last_actual) * 100, 2)
                else:
                    accuracy = None
                    last_actual = None

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(years, yields, marker='o', color='b', 
                    label=f'{crop} Yield over the Years')
                plt.title(f'{crop} Yield Trend')
                plt.xlabel('Year')
                plt.ylabel('Yield (hg/ha)')
                plt.grid(True)
                plt.legend()

                # Save plot to memory
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')

                # Clean up matplotlib resources
                plt.close('all')

                return render_template("result.html",
                                   prediction=prediction,
                                   crop=crop,
                                   area=area,
                                   years=years,
                                   yields=yields,
                                   avg_yield=avg_yield,
                                   max_yield=max_yield,
                                   min_yield=min_yield,
                                   std_yield=std_yield,
                                   growth_rate=growth_rate,
                                   recent_trend=recent_trend,
                                   pred_vs_avg=pred_vs_avg,
                                   pred_vs_last=pred_vs_last,
                                   actual_yields=actual_yields,
                                   actual_years=actual_years,
                                   accuracy=accuracy,
                                   last_actual=last_actual,
                                   plot_url=plot_url,
                                   year=year)

            except Exception as e:
                plt.close('all')  # Clean up in case of error
                return render_template("index.html", 
                    prediction_text=f"Error processing data: {str(e)}")

        except Exception as e:
            return render_template("index.html", 
                prediction_text=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
