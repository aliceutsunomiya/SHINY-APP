from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and preprocess data
data = pd.read_csv('Lotwize Data with New Features (1).csv')

# Preprocessing
numerical_columns = ['bathrooms', 'bedrooms', 'livingArea', 'lotSize', 'yearBuilt', 
                    'NearestSchoolDistance', 'NearestHospitalDistance', 'ShopsIn2Miles']
categorical_columns = ['city', 'region', 'homeType']

# Imputation
for col in numerical_columns:
    data[col] = data[col].fillna(data[col].median())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Handle outliers
Q1 = data[numerical_columns].quantile(0.25)
Q3 = data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for col in numerical_columns:
    data[col] = data[col].clip(lower=lower_bound[col], upper=upper_bound[col])

# Encode categorical features
data_encoded = pd.get_dummies(data.drop(columns=['monthSold', 'city']), 
                            columns=categorical_columns[1:], 
                            drop_first=True)

# Define target and features
X = data_encoded.drop(columns=['price'])
y = data_encoded['price']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Function to calculate feature contributions
def calculate_feature_contributions(model, input_df, feature_names):
    # Get coefficients and calculate base value (intercept)
    base_value = model.intercept_
    
    # Calculate contribution for each feature
    contributions = []
    for i, feature in enumerate(feature_names):
        contribution = model.coef_[i] * input_df.iloc[0][feature]
        contributions.append({
            'feature': feature,
            'contribution': contribution
        })
    
    # Sort contributions by absolute value
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return base_value, contributions

# Define the UI
app_ui = ui.page_fluid(
    ui.tags.style("""
        /* Global styles */
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .container-fluid {
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .well {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .form-control {
            border-radius: 6px;
            border: 1px solid #ddd;
            padding: 8px 12px;
            margin-bottom: 15px;
            width: 100%;
        }
        .form-control:focus {
            border-color: #2c3e50;
            box-shadow: 0 0 0 0.2rem rgba(44, 62, 80, 0.25);
        }
        .btn-primary {
            background-color: #2c3e50;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            transition: all 0.3s;
            width: 100%;
            font-size: 16px;
            font-weight: 500;
        }
        .btn-primary:hover {
            background-color: #34495e;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-weight: 600;
            font-size: 24px;
        }
        h4 {
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 500;
            font-size: 18px;
        }
        .prediction-panel {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 200px;
            text-align: center;
        }
        .input-label {
            font-weight: 500;
            color: #445566;
            margin-bottom: 5px;
        }
        .app-header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            text-align: center;
        }
        .prediction-result {
            font-size: 24px;
            font-weight: 600;
            color: #2c3e50;
            margin: 20px 0;
        }
        .prediction-details {
            color: #666;
            font-size: 16px;
            margin-top: 10px;
        }
        .feature-icon {
            margin-right: 8px;
            color: #2c3e50;
        }
    """),
    
    ui.div(
        {"class": "app-header"},
        ui.h2("🏠 Real Estate Price Predictor", style="color: white; margin: 0;"),
        ui.p("Enter house details below to get an estimated price", style="margin: 10px 0 0 0;")
    ),
    
    ui.row(
        ui.column(4,
            ui.div(
                {"class": "well"},
                ui.h2("House Features"),
                
                ui.h4("📊 Basic Information"),
                ui.input_numeric("bedrooms", "Number of Bedrooms", value=3, min=1, max=10),
                ui.input_numeric("bathrooms", "Number of Bathrooms", value=2, min=1, max=10),
                ui.input_numeric("livingArea", "Living Area (sq ft)", value=2000, min=500, max=10000),
                ui.input_numeric("lotSize", "Lot Size (sq ft)", value=5000, min=1000, max=50000),
                
                ui.h4("🏠 Property Details"),
                ui.input_numeric("yearBuilt", "Year Built", value=2000, min=1900, max=2024),
                ui.input_select("region", "Region", choices=list(data['region'].unique())),
                ui.input_select("homeType", "Home Type", choices=list(data['homeType'].unique())),
                
                ui.h4("📍 Location Features"),
                ui.input_numeric("NearestSchoolDistance", "Distance to School (miles)", 
                               value=1, min=0, max=10),
                ui.input_numeric("NearestHospitalDistance", "Distance to Hospital (miles)", 
                               value=2, min=0, max=20),
                ui.input_numeric("ShopsIn2Miles", "Shops within 2 Miles", 
                               value=10, min=0, max=100),
                
                ui.div(
                    {"style": "margin-top: 30px;"},
                    ui.input_action_button(
                        "predict", "🔍 Calculate Price Prediction",
                        class_="btn-primary"
                    )
                )
            )
        ),
        ui.column(8,
            ui.div(
                {"class": "prediction-panel"},
                ui.h2("Prediction Results"),
                ui.div(
                    {"class": "prediction-result"},
                    ui.output_text("prediction")
                ),
                ui.div(
                    {"class": "prediction-details"},
                    ui.output_text("prediction_details")
                ),
                ui.h3("Feature Importance Breakdown", 
                     style="margin-top: 30px; margin-bottom: 20px;"),
                ui.output_plot("waterfall_chart", height="500px")
            )
        )
    )
)

def server(input, output, session):
    # Store model prediction and data for reactive use
    prediction_store = reactive.Value({
        'price': None, 
        'error': None,
        'base_value': None,
        'contributions': None
    })
    
    @reactive.Effect
    @reactive.event(input.predict)
    def predict():
        try:
            # Prepare input data
            input_data = {
                'bedrooms': input.bedrooms(),
                'bathrooms': input.bathrooms(),
                'livingArea': input.livingArea(),
                'lotSize': input.lotSize(),
                'yearBuilt': input.yearBuilt(),
                'NearestSchoolDistance': input.NearestSchoolDistance(),
                'NearestHospitalDistance': input.NearestHospitalDistance(),
                'ShopsIn2Miles': input.ShopsIn2Miles(),
            }
            
            # Add dummy variables
            for region in X_train.filter(regex='^region_').columns:
                region_name = region.replace('region_', '')
                input_data[region] = 1 if input.region() == region_name else 0
                
            for home_type in X_train.filter(regex='^homeType_').columns:
                type_name = home_type.replace('homeType_', '')
                input_data[home_type] = 1 if input.homeType() == type_name else 0
            
            # Create DataFrame with aligned features
            input_df = pd.DataFrame([{col: input_data.get(col, 0) for col in X_train.columns}])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Calculate feature contributions
            base_value, contributions = calculate_feature_contributions(
                model, input_df, X_train.columns
            )
            
            prediction_store.set({
                'price': prediction,
                'error': None,
                'base_value': base_value,
                'contributions': contributions
            })
            
        except Exception as e:
            prediction_store.set({
                'price': None,
                'error': str(e),
                'base_value': None,
                'contributions': None
            })
    
    @output
    @render.text
    def prediction():
        pred = prediction_store.get()
        if pred['error']:
            return f"❌ Error: {pred['error']}"
        elif pred['price'] is not None:
            return f"💰 Estimated Price: ${pred['price']:,.2f}"
        return "👋 Enter house details and click 'Calculate Price Prediction'"
    
    @output
    @render.text
    def prediction_details():
        pred = prediction_store.get()
        if pred['price'] is not None:
            return """Based on current market conditions and comparable properties
                     Prediction confidence may vary based on market volatility"""
        return ""
    
    @output
    @render.plot
    def waterfall_chart():
        pred = prediction_store.get()
        if pred['contributions'] is None:
            return None
            
        # Prepare data
        base_value = pred['base_value']
        contributions = pred['contributions']
        
        # Filter and group small contributions
        threshold = 1000  # $1000 minimum contribution
        significant_contributions = []
        other_contribution = 0
        
        for c in contributions:
            if abs(c['contribution']) >= threshold:
                # Clean up feature names
                feature_name = c['feature']
                # Handle region features
                if feature_name.startswith('region_'):
                    feature_name = 'Region: ' + feature_name.replace('region_', '')
                # Handle home type features
                elif feature_name.startswith('homeType_'):
                    feature_name = 'Type: ' + feature_name.replace('homeType_', '')
                # Shorten other feature names
                else:
                    feature_map = {
                        'NearestSchoolDistance': 'School Dist.',
                        'NearestHospitalDistance': 'Hospital Dist.',
                        'ShopsIn2Miles': 'Nearby Shops',
                        'livingArea': 'Living Area',
                        'lotSize': 'Lot Size',
                        'yearBuilt': 'Year Built',
                        'bedrooms': 'Bedrooms',
                        'bathrooms': 'Bathrooms'
                    }
                    feature_name = feature_map.get(feature_name, feature_name)
                
                significant_contributions.append({
                    'feature': feature_name,
                    'contribution': c['contribution']
                })
            else:
                other_contribution += c['contribution']
        
        # Sort by absolute contribution value and take top 10
        significant_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        significant_contributions = significant_contributions[:10]
        
        # Add "Other" category if significant
        if abs(other_contribution) >= threshold:
            significant_contributions.append({
                'feature': 'Other Features',
                'contribution': other_contribution
            })
        
        # Create lists for plotting - CORRECTED ORDER
        labels = ['Base Value']
        values = [base_value]
        running_total = base_value
        
        # Add intermediate values
        for c in significant_contributions:
            labels.append(c['feature'])
            values.append(c['contribution'])
            running_total += c['contribution']
        
        # Add final prediction
        labels.append('Final Prediction')
        values.append(pred['price'] - running_total)
        
        # Calculate positions for the waterfall
        pos = np.zeros(len(values))
        pos[0] = values[0]
        for i in range(1, len(values)-1):
            pos[i] = pos[i-1] + values[i]
        pos[-1] = pred['price']
        
        # Create figure and axis with adjusted size ratio
        fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
        plt.subplots_adjust(bottom=0.3, top=0.85, left=0.1, right=0.95)
        
        # Set clean background
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('white')
        
        # Define colors
        positive_color = '#2E86C1'  # Darker blue for positive values
        negative_color = '#E74C3C'  # Red for negative values
        base_color = '#2C3E50'      # Dark blue-gray for base and final
        connection_color = '#95A5A6' # Gray for connecting lines
        
        # Plot connecting lines first (before bars)
        for i in range(len(values)-1):
            if i == 0:
                start_y = values[i]
            else:
                start_y = pos[i]
            
            if i == len(values)-2:
                end_y = pos[-1]
            else:
                end_y = pos[i+1]
                
            ax.plot([i+0.35, i+0.65], [start_y, end_y], 
                    color=connection_color, linestyle='--', alpha=0.5)
        
        # Plot bars with color coding
        bar_width = 0.7
        for i in range(len(values)):
            if i == 0 or i == len(values)-1:
                color = base_color
            else:
                color = positive_color if values[i] >= 0 else negative_color
                
            if i == 0:
                ax.bar(i, values[i], bottom=0, color=color, width=bar_width)
            elif i == len(values)-1:
                ax.bar(i, values[i], bottom=0, color=color, width=bar_width)
            else:
                ax.bar(i, values[i], bottom=pos[i]-values[i], color=color, width=bar_width)
        
        # Style customization
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Enhanced grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='#666666')
        ax.set_axisbelow(True)
        
        # Customize ticks
        ax.tick_params(axis='both', colors='#333333', length=5)
        
        # Set x-axis labels with better formatting
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            labels,
            rotation=30,
            ha='right',
            rotation_mode='anchor',
            fontsize=11,
            fontweight='medium'
        )
        
        # Format y-axis with cleaner labels
        def format_currency(x, p):
            if abs(x) >= 1e6:
                return f'${x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'${x/1e3:.0f}K'
            else:
                return f'${x:.0f}'
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_currency))
        
        # Enhanced title and labels
        plt.suptitle('Home Price Feature Importance', 
                    fontsize=16, 
                    fontweight='bold', 
                    color='#2C3E50',
                    y=0.95)
        
        plt.title('How Different Features Affect the Final Price Prediction', 
                pad=20,
                fontsize=12,
                color='#666666',
                style='italic')
        
        ax.set_ylabel('Price Contribution', 
                    fontsize=12, 
                    color='#2C3E50',
                    fontweight='medium',
                    labelpad=10)
        
        # Add value labels with improved positioning and formatting
        for i, v in enumerate(values):
            if i == 0:
                y = v
            elif i == len(values)-1:
                y = v
            else:
                y = pos[i]
            
            if abs(v) >= 1e6:
                value_text = f'${v/1e6:.1f}M'
            elif abs(v) >= 1e3:
                value_text = f'${v/1e3:.0f}K'
            else:
                value_text = f'${v:.0f}'
                
            # Position labels above for positive values, below for negative
            va = 'bottom' if v >= 0 else 'top'
            y_offset = 0.01 * max(abs(min(values)), abs(max(values)))
            y_pos = y + y_offset if v >= 0 else y - y_offset
            
            # Add +/- signs for feature contributions
            if i not in [0, len(values)-1]:  # Skip base and final values
                value_text = f'+{value_text}' if v >= 0 else f'-{value_text.replace("-", "")}'
            
            ax.text(i, y_pos,
                    value_text,
                    ha='center',
                    va=va,
                    fontsize=10,
                    fontweight='medium',
                    color='#333333')
        
        # Final layout adjustments
        plt.tight_layout(rect=[0.05, 0.25, 0.95, 0.90])  # Increased bottom margin
        ax.margins(y=0.2)
        
        return fig

app = App(app_ui, server)

if __name__ == "__main__":
    app.run(launch_browser=True)



