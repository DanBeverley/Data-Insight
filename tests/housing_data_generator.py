import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
import uuid
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class HousingDataGenerator:
    def __init__(self, n_samples: int = 5000, random_state: int = 42):
        self.n_samples = n_samples
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define realistic housing data distributions
        self.cities = {
            'New York': {'base_price': 650000, 'price_std': 200000, 'zip_range': (10001, 10299)},
            'Los Angeles': {'base_price': 550000, 'price_std': 180000, 'zip_range': (90001, 90299)},
            'Chicago': {'base_price': 320000, 'price_std': 120000, 'zip_range': (60601, 60699)},
            'Houston': {'base_price': 280000, 'price_std': 100000, 'zip_range': (77001, 77099)},
            'Phoenix': {'base_price': 350000, 'price_std': 110000, 'zip_range': (85001, 85099)},
            'Philadelphia': {'base_price': 290000, 'price_std': 95000, 'zip_range': (19101, 19199)},
            'San Antonio': {'base_price': 250000, 'price_std': 80000, 'zip_range': (78201, 78299)},
            'San Diego': {'base_price': 700000, 'price_std': 220000, 'zip_range': (92101, 92199)},
            'Dallas': {'base_price': 320000, 'price_std': 115000, 'zip_range': (75201, 75299)},
            'San Jose': {'base_price': 950000, 'price_std': 300000, 'zip_range': (95101, 95199)}
        }
        
        self.property_types = ['Single Family', 'Townhouse', 'Condo', 'Multi-Family', 'Mobile Home']
        self.home_styles = ['Ranch', 'Colonial', 'Contemporary', 'Victorian', 'Split Level', 'Cape Cod', 'Tudor', 'Modern']
        self.heating_types = ['Gas', 'Electric', 'Oil', 'Heat Pump', 'Solar', 'Geothermal']
        self.cooling_types = ['Central Air', 'Window Units', 'Heat Pump', 'Evaporative', 'None']
        
    def generate_housing_dataset(self) -> pd.DataFrame:
        """Generate comprehensive housing dataset"""
        
        # Base property information
        df = self._generate_basic_property_info()
        
        # Location and geographic data
        df = self._add_location_data(df)
        
        # Physical characteristics
        df = self._add_physical_characteristics(df)
        
        # Financial and market data
        df = self._add_financial_data(df)
        
        # Amenities and features
        df = self._add_amenities_and_features(df)
        
        # Historical and temporal data
        df = self._add_temporal_data(df)
        
        # Market conditions and trends
        df = self._add_market_conditions(df)
        
        # Quality issues and missing data
        df = self._introduce_data_quality_issues(df)
        
        # PII and sensitive information
        df = self._add_pii_and_agent_data(df)
        
        # Target variables for different ML tasks
        df = self._add_target_variables(df)
        
        return df
    
    def _generate_basic_property_info(self) -> pd.DataFrame:
        """Generate basic property information"""
        
        # Property IDs
        property_ids = [f"PROP_{str(uuid.uuid4())[:8].upper()}" for _ in range(self.n_samples)]
        
        # MLS numbers (realistic format)
        mls_numbers = [f"MLS{np.random.randint(1000000, 9999999)}" for _ in range(self.n_samples)]
        
        # Property types with realistic distribution
        property_types = np.random.choice(
            self.property_types, 
            self.n_samples, 
            p=[0.65, 0.15, 0.12, 0.06, 0.02]
        )
        
        # Home styles
        home_styles = np.random.choice(self.home_styles, self.n_samples)
        
        df = pd.DataFrame({
            'property_id': property_ids,
            'mls_number': mls_numbers,
            'property_type': property_types,
            'home_style': home_styles
        })
        
        return df
    
    def _add_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location and geographic information"""
        
        # Cities with realistic distribution (some cities more common)
        cities = np.random.choice(
            list(self.cities.keys()), 
            len(df), 
            p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.10, 0.11, 0.10]
        )
        
        # Generate addresses
        street_names = ['Main St', 'Oak Ave', 'Park Blvd', 'Elm St', 'Maple Dr', 'Cedar Ln', 'Pine St', 'Washington Ave']
        addresses = []
        zip_codes = []
        neighborhoods = []
        
        for city in cities:
            street_number = np.random.randint(100, 9999)
            street_name = np.random.choice(street_names)
            address = f"{street_number} {street_name}"
            addresses.append(address)
            
            # Generate zip code based on city
            zip_range = self.cities[city]['zip_range']
            zip_code = np.random.randint(zip_range[0], zip_range[1])
            zip_codes.append(str(zip_code))
            
            # Generate neighborhood names
            neighborhood_types = ['Heights', 'Park', 'Grove', 'Hills', 'Gardens', 'Plaza', 'Village']
            neighborhood = f"{np.random.choice(['North', 'South', 'East', 'West', 'Central'])} {np.random.choice(neighborhood_types)}"
            neighborhoods.append(neighborhood)
        
        # School districts (simplified)
        school_districts = [f"{city} ISD District {np.random.randint(1, 15)}" for city in cities]
        
        # School ratings (1-10 scale)
        school_ratings = np.random.choice(range(1, 11), len(df), p=[0.05, 0.05, 0.08, 0.12, 0.15, 0.20, 0.15, 0.10, 0.07, 0.03])
        
        df['city'] = cities
        df['address'] = addresses
        df['zip_code'] = zip_codes
        df['neighborhood'] = neighborhoods
        df['school_district'] = school_districts
        df['school_rating'] = school_ratings
        
        return df
    
    def _add_physical_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add physical property characteristics"""
        
        # Bedrooms (realistic distribution)
        bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], len(df), p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
        
        # Bathrooms (correlated with bedrooms)
        bathrooms = []
        for bed in bedrooms:
            if bed == 1:
                bath = np.random.choice([1, 1.5], p=[0.7, 0.3])
            elif bed == 2:
                bath = np.random.choice([1, 1.5, 2], p=[0.2, 0.3, 0.5])
            elif bed == 3:
                bath = np.random.choice([1.5, 2, 2.5], p=[0.2, 0.6, 0.2])
            elif bed == 4:
                bath = np.random.choice([2, 2.5, 3], p=[0.3, 0.5, 0.2])
            else:  # 5+ bedrooms
                bath = np.random.choice([2.5, 3, 3.5, 4], p=[0.3, 0.4, 0.2, 0.1])
            bathrooms.append(bath)
        
        # Square footage (correlated with bedrooms)
        base_sqft = bedrooms * 400 + np.random.normal(500, 200, len(df))
        square_footage = np.maximum(base_sqft, 500).astype(int)
        
        # Lot size (in acres, converted to square feet for some)
        lot_sizes_acres = np.random.lognormal(np.log(0.25), 0.8, len(df))
        lot_sizes_sqft = lot_sizes_acres * 43560  # Convert to square feet
        
        # Year built (realistic distribution)
        current_year = datetime.now().year
        year_built = np.random.choice(
            range(1920, current_year + 1),
            len(df),
            p=self._generate_year_built_distribution(1920, current_year)
        )
        
        # Stories
        stories = np.random.choice([1, 1.5, 2, 2.5, 3], len(df), p=[0.45, 0.15, 0.25, 0.10, 0.05])
        
        # Garage information
        garage_spaces = np.random.choice([0, 1, 2, 3, 4], len(df), p=[0.15, 0.20, 0.45, 0.15, 0.05])
        garage_types = np.random.choice(['None', 'Attached', 'Detached', 'Carport'], len(df), p=[0.15, 0.60, 0.20, 0.05])
        
        # HVAC systems
        heating_systems = np.random.choice(self.heating_types, len(df))
        cooling_systems = np.random.choice(self.cooling_types, len(df))
        
        df['bedrooms'] = bedrooms
        df['bathrooms'] = bathrooms
        df['square_footage'] = square_footage
        df['lot_size_acres'] = np.round(lot_sizes_acres, 3)
        df['lot_size_sqft'] = lot_sizes_sqft.astype(int)
        df['year_built'] = year_built
        df['stories'] = stories
        df['garage_spaces'] = garage_spaces
        df['garage_type'] = garage_types
        df['heating_system'] = heating_systems
        df['cooling_system'] = cooling_systems
        
        return df
    
    def _generate_year_built_distribution(self, start_year: int, end_year: int) -> List[float]:
        """Generate realistic year built distribution"""
        years = list(range(start_year, end_year + 1))
        
        # More houses built in recent decades, with construction booms
        weights = []
        for year in years:
            if year < 1950:
                weight = 0.3
            elif year < 1970:
                weight = 0.8
            elif year < 1990:
                weight = 1.2
            elif year < 2000:
                weight = 1.5
            elif year < 2010:
                weight = 1.8
            elif year < 2020:
                weight = 1.2
            else:
                weight = 0.8
            weights.append(weight)
        
        # Normalize to probabilities
        total_weight = sum(weights)
        return [w / total_weight for w in weights]
    
    def _add_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add financial and pricing information"""
        
        # Base price calculation based on city, size, and age
        prices = []
        for idx, row in df.iterrows():
            city_data = self.cities[row['city']]
            base_price = city_data['base_price']
            price_std = city_data['price_std']
            
            # Size adjustment
            size_multiplier = (row['square_footage'] / 2000) ** 0.7
            
            # Age adjustment (newer homes cost more, with depreciation)
            age = datetime.now().year - row['year_built']
            if age < 5:
                age_multiplier = 1.2
            elif age < 15:
                age_multiplier = 1.1
            elif age < 30:
                age_multiplier = 1.0
            elif age < 50:
                age_multiplier = 0.9
            else:
                age_multiplier = 0.7
            
            # Quality adjustment based on features
            quality_multiplier = 1.0
            if row['garage_spaces'] >= 2:
                quality_multiplier += 0.1
            if row['bathrooms'] >= 3:
                quality_multiplier += 0.1
            if row['school_rating'] >= 8:
                quality_multiplier += 0.15
            
            price = base_price * size_multiplier * age_multiplier * quality_multiplier
            price += np.random.normal(0, price_std * 0.3)
            prices.append(max(price, 50000))  # Minimum price
        
        # Property taxes (as percentage of home value)
        property_tax_rates = np.random.normal(0.012, 0.005, len(df))
        property_tax_rates = np.clip(property_tax_rates, 0.005, 0.035)
        annual_property_tax = [price * rate for price, rate in zip(prices, property_tax_rates)]
        
        # HOA fees (not all properties have them)
        has_hoa = np.random.choice([True, False], len(df), p=[0.35, 0.65])
        hoa_fees = []
        for has_fee in has_hoa:
            if has_fee:
                fee = np.random.lognormal(np.log(150), 0.8)
                hoa_fees.append(round(fee, 2))
            else:
                hoa_fees.append(0.0)
        
        # Days on market
        days_on_market = np.random.gamma(2, 15, len(df)).astype(int)
        
        # Price per square foot
        price_per_sqft = [price / sqft for price, sqft in zip(prices, df['square_footage'])]
        
        df['listing_price'] = [round(price, -3) for price in prices]  # Round to nearest thousand
        df['property_tax_rate'] = np.round(property_tax_rates, 4)
        df['annual_property_tax'] = [round(tax, 2) for tax in annual_property_tax]
        df['hoa_fee_monthly'] = hoa_fees
        df['days_on_market'] = days_on_market
        df['price_per_sqft'] = [round(ppsf, 2) for ppsf in price_per_sqft]
        
        return df
    
    def _add_amenities_and_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add amenities and special features"""
        
        # Boolean features with realistic probabilities
        features = {
            'has_pool': 0.15,
            'has_spa': 0.08,
            'has_fireplace': 0.35,
            'has_basement': 0.25,
            'has_attic': 0.60,
            'has_balcony': 0.20,
            'has_deck_patio': 0.55,
            'has_garden': 0.30,
            'hardwood_floors': 0.40,
            'updated_kitchen': 0.35,
            'updated_bathrooms': 0.30,
            'new_roof': 0.20,
            'new_hvac': 0.25,
            'energy_efficient': 0.40,
            'smart_home_features': 0.25,
            'security_system': 0.30,
            'gated_community': 0.15
        }
        
        for feature, probability in features.items():
            df[feature] = np.random.choice([True, False], len(df), p=[probability, 1-probability])
        
        # Appliances included
        appliances = []
        appliance_options = ['Refrigerator', 'Dishwasher', 'Washer/Dryer', 'Microwave', 'Oven/Range']
        for _ in range(len(df)):
            num_appliances = np.random.randint(0, len(appliance_options) + 1)
            included = random.sample(appliance_options, num_appliances)
            appliances.append(', '.join(included) if included else 'None')
        
        df['appliances_included'] = appliances
        
        # Parking information
        parking_types = ['Street', 'Driveway', 'Garage', 'Covered', 'Assigned Spot']
        df['parking_type'] = np.random.choice(parking_types, len(df))
        
        return df
    
    def _add_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal information"""
        
        # Listing dates (within last 2 years)
        base_date = datetime.now() - timedelta(days=730)
        listing_dates = [
            base_date + timedelta(days=int(np.random.exponential(180)))
            for _ in range(len(df))
        ]
        
        # Last sold dates (some properties)
        last_sold_dates = []
        last_sold_prices = []
        
        for idx, listing_date in enumerate(listing_dates):
            if np.random.random() < 0.7:  # 70% have previous sale data
                years_back = np.random.exponential(5)
                sold_date = listing_date - timedelta(days=int(years_back * 365))
                last_sold_dates.append(sold_date)
                
                # Previous price (typically lower due to appreciation)
                current_price = df.iloc[idx]['listing_price']
                appreciation_rate = np.random.normal(0.05, 0.03)  # 5% annual appreciation
                years_diff = years_back
                previous_price = current_price / ((1 + appreciation_rate) ** years_diff)
                last_sold_prices.append(round(previous_price, -3))
            else:
                last_sold_dates.append(None)
                last_sold_prices.append(None)
        
        df['listing_date'] = listing_dates
        df['last_sold_date'] = last_sold_dates
        df['last_sold_price'] = last_sold_prices
        
        return df
    
    def _add_market_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market condition indicators"""
        
        # Market trends
        market_trends = np.random.choice(['Hot', 'Warm', 'Neutral', 'Cool', 'Cold'], 
                                       len(df), p=[0.2, 0.25, 0.3, 0.15, 0.1])
        
        # Neighborhood price trends
        price_trends = np.random.choice(['Rising', 'Stable', 'Declining'], 
                                      len(df), p=[0.45, 0.40, 0.15])
        
        # Competition (number of similar listings nearby)
        nearby_listings = np.random.poisson(8, len(df))
        
        # Walk score (walkability rating)
        walk_scores = np.random.choice(range(0, 101), len(df))
        
        # Transit score
        transit_scores = np.random.choice(range(0, 101), len(df))
        
        df['market_trend'] = market_trends
        df['neighborhood_price_trend'] = price_trends
        df['nearby_listings'] = nearby_listings
        df['walk_score'] = walk_scores
        df['transit_score'] = transit_scores
        
        return df
    
    def _introduce_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Introduce realistic data quality issues"""
        
        # Missing values with patterns
        # Older properties more likely to have missing renovation data
        older_mask = df['year_built'] < 1980
        df.loc[older_mask & (np.random.random(len(df)) < 0.3), 'new_roof'] = None
        df.loc[older_mask & (np.random.random(len(df)) < 0.3), 'new_hvac'] = None
        
        # Some properties missing last sold information
        missing_sold_mask = np.random.random(len(df)) < 0.1
        df.loc[missing_sold_mask, 'last_sold_date'] = None
        df.loc[missing_sold_mask, 'last_sold_price'] = None
        
        # Random missing values in other columns
        for col in ['hoa_fee_monthly', 'walk_score', 'transit_score']:
            missing_mask = np.random.random(len(df)) < 0.05
            df.loc[missing_mask, col] = None
        
        # Data entry errors
        # Inconsistent property types
        error_mask = np.random.random(len(df)) < 0.02
        df.loc[error_mask, 'property_type'] = df.loc[error_mask, 'property_type'].str.replace('Single Family', 'SFH')
        
        # Outliers in financial data
        outlier_mask = np.random.random(len(df)) < 0.01
        df.loc[outlier_mask, 'listing_price'] *= np.random.uniform(2, 5, outlier_mask.sum())
        
        return df
    
    def _add_pii_and_agent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add agent and contact information (PII)"""
        
        # Real estate agent information
        agent_names = [
            'John Smith', 'Sarah Johnson', 'Michael Brown', 'Emily Davis', 'David Wilson',
            'Lisa Garcia', 'Robert Miller', 'Jennifer Taylor', 'Christopher Anderson', 'Amanda Martinez'
        ]
        
        agents = np.random.choice(agent_names, len(df))
        
        # Agent phone numbers
        agent_phones = []
        for _ in range(len(df)):
            phone = f"({np.random.randint(200,999)}) {np.random.randint(200,999)}-{np.random.randint(1000,9999)}"
            agent_phones.append(phone)
        
        # Agent emails
        agent_emails = []
        for agent in agents:
            first_name = agent.split()[0].lower()
            last_name = agent.split()[1].lower()
            domain = np.random.choice(['realty.com', 'homes.com', 'realtor.net'])
            email = f"{first_name}.{last_name}@{domain}"
            agent_emails.append(email)
        
        # Listing agent license numbers
        license_numbers = [f"LIC{np.random.randint(100000, 999999)}" for _ in range(len(df))]
        
        df['listing_agent'] = agents
        df['agent_phone'] = agent_phones
        df['agent_email'] = agent_emails
        df['agent_license'] = license_numbers
        
        return df
    
    def _add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variables for different ML tasks"""
        
        # Regression: Price prediction (already have listing_price)
        # Classification: Price category
        price_categories = pd.cut(df['listing_price'], 
                                bins=[0, 200000, 400000, 600000, 1000000, float('inf')],
                                labels=['Budget', 'Mid-Range', 'Upper-Mid', 'Luxury', 'Ultra-Luxury'])
        
        # Binary classification: Quick sale (sells within 30 days)
        quick_sale_probability = (
            0.3 +  # Base probability
            (100 - df['days_on_market']) / 200 * 0.4 +  # Days on market impact
            (df['school_rating'] / 10) * 0.2 +  # School rating impact
            (df['price_per_sqft'] < df['price_per_sqft'].median()) * 0.1  # Competitive pricing
        )
        quick_sale = np.random.binomial(1, np.clip(quick_sale_probability, 0, 1), len(df))
        
        # Multi-class: Market segment
        market_segments = []
        for _, row in df.iterrows():
            if row['property_type'] == 'Condo' and row['listing_price'] < 300000:
                segment = 'First-Time Buyer'
            elif row['bedrooms'] >= 4 and row['listing_price'] > 500000:
                segment = 'Family Premium'
            elif row['year_built'] < 1960 and row['listing_price'] > 400000:
                segment = 'Historic Premium'
            elif row['bedrooms'] <= 2 and row['listing_price'] < 400000:
                segment = 'Starter Home'
            else:
                segment = 'General Market'
            market_segments.append(segment)
        
        df['price_category'] = price_categories
        df['quick_sale'] = quick_sale
        df['market_segment'] = market_segments
        
        return df

def generate_housing_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """Generate comprehensive housing dataset"""
    generator = HousingDataGenerator(n_samples=n_samples)
    dataset = generator.generate_housing_dataset()
    
    print(f"Generated housing dataset with {len(dataset)} rows and {len(dataset.columns)} columns")
    print(f"Dataset size: {dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print("\nColumn types:")
    print(dataset.dtypes.value_counts())
    print(f"\nMissing values: {dataset.isnull().sum().sum()}")
    print(f"Duplicate rows: {dataset.duplicated().sum()}")
    
    # Show sample statistics
    print(f"\nPrice range: ${dataset['listing_price'].min():,.0f} - ${dataset['listing_price'].max():,.0f}")
    print(f"Average price: ${dataset['listing_price'].mean():,.0f}")
    print(f"Square footage range: {dataset['square_footage'].min()} - {dataset['square_footage'].max()} sq ft")
    
    return dataset

if __name__ == "__main__":
    # Generate housing dataset
    df = generate_housing_dataset(5000)
    
    # Save to file
    output_path = "/mnt/c/Users/ADMIN/Documents/GitHub/Data-Insight/tests/housing_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nHousing dataset saved to: {output_path}")
    
    # Display sample
    print("\nSample data:")
    print(df.head())
    
    print("\nDataset summary:")
    print(df[['listing_price', 'square_footage', 'bedrooms', 'bathrooms', 'year_built']].describe())