# EE104_Super_Project_Amaton_Charles_Karim
# **EE_104-project-1**

## Access Google Sheet for Project:
```
https://docs.google.com/spreadsheets/d/1A244oU6kBScmfyDnN1Il7ojfzGIi3WEoOAdTkmwWnvc/edit#gid=0
```

## Create a Virtual Environment:
```
py -m venv env
.\env\Scripts\activate
```

## For Installing Libraries Used in Project:
```
pip install -r requirements.txt
```
## To Run Program:
```
py main.py
```


## Instructions:

### _Main Function:_
- Give User option to either book a passenger for a flight or commence/continue boarding for a flight
  - If booking is chosen, main will have user input a flight number, passenger name, and seat number
  - If boarding option is chosen, take user to boarding function
  - If user wants to quit, enter q

### _For Booking (Sabeeq Karim):_
- Checks if flight number/ worksheet for flight exists
  - If flight does not exist, a new worksheet will be created with according flight number entered
- Pull worksheet into pandas DataFrame
  - If worksheet is new/empty, set width and row of DataFrame
    - Columns for seat, name, and boarding status
    - For the project, kept it set for 20 rows to keep it simple
- Check if name inputted by user exists in name column
  - checks if inputted name does not exist
    - if name exists, then prints statement saying so and returns to main
- Check if seat number inputted by user is in seat column
  - checks if seat number already taken by another person
    - if seat number already taken, then prints statement saying so and returns to main
- If all if statements met:
  - add the row with the inputted name and seat number along with False boolean for boarding status
  - update according worksheet from pandas DataFrame

### _For Boarding (Sebastian Charles):_
- user inputs number of flight that they want to start boarding
  - import all worksheets and check if flight number exists in any of the sheet sheet names
    - if not, print statement of flight number not existing and return to main
- If flight exists, then import according worksheet from Google sheets
- Import selected worksheet into pandas DataFrame
- print Data Frame to give user current look of worksheet
- convert any boarding statuses from strings to booleans
- Check if boarding for worksheet is complete
  - Check if all boarding boolean values are True
    - If so, then print that flight has already completed boarding and return to main
- Start while loop to stay in same worksheet until user chooses to quit
  - Ask user to input a name
    - Checks in name column if name entered exists
      - If not, print that name does not exist and return to main
  - If name exists, ask user to input their according seat number
    - Checks seat number for particular row for the inputted name
      - If seat number is not in row for according passenger, print that the seat number does not match info for according passenger and return to main
  - If seat is also correctly matched then get according row from index in Data Frame
  - Check if boarding boolean value for specific row is True
    - If boarding cell value is True then print saying that passenger has already boarded and return to main
  - If boarding boolean value is False, then change value to True and print that the Passenger is being boarded
  - Export updated Data Frame to according Google Sheets worksheet
  - Check if all boarding boolean values are True
    - If so, print that the flight has now completed boarding and return to main
  - If not all boarding boolean values are True, ask user if they want to continue boarding by entering yes or no
    - If user enters no, then break out of while loop
    - If user enters yes, then start rerun loop again


## Resources:
```
site with all kinds dataframe functions: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html
site with all gspread functions: https://gspread.readthedocs.io/en/latest/user-guide.html#

```
