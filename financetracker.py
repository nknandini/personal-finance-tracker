import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load or create dataset
try:
    df = pd.read_csv("finance_tracker.csv")
except:
    df = pd.DataFrame(columns=["Date", "Category", "Amount", "Description"])


# ------------------- ADD EXPENSE -------------------
def add_expense():
    global df

    date = input("Enter date (YYYY-MM-DD): ")
    category = input("Category (Food, Travel, etc): ")

    try:
        amount = float(input("Amount: "))
    except:
        print("❌ Invalid amount")
        return

    description = input("Description: ")

    new_row = pd.DataFrame([[date, category, amount, description]],
                           columns=df.columns)

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("finance_tracker.csv", index=False)

    print("✅ Expense added!")


# ------------------- VIEW SUMMARY -------------------
def view_summary():
    print("\n💰 Total Spending:", df["Amount"].sum())

    print("\n📊 Category-wise spending:")
    print(df.groupby("Category")["Amount"].sum().to_string())

    check_budget()
    highest_spending_category()


# ------------------- BUDGET CHECK -------------------
BUDGET = 10000

def check_budget():
    total = df["Amount"].sum()
    if total > BUDGET:
        print(f"\n⚠️ You crossed your budget! Overspent by {total - BUDGET:.2f}")
    else:
        print(f"\n✅ Remaining budget: {BUDGET - total:.2f}")


# ------------------- CATEGORY INSIGHT -------------------
def highest_spending_category():
    if len(df) == 0:
        return
    category = df.groupby("Category")["Amount"].sum().idxmax()
    print(f"\n💡 You spend the most on: {category}")


# ------------------- MONTHLY SUMMARY -------------------
def monthly_summary():
    if len(df) == 0:
        print("No data available.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M")

    monthly = df.groupby("Month")["Amount"].sum()
    print("\n📅 Monthly Spending:")
    print(monthly.to_string())


# ------------------- GRAPH -------------------
def show_graph():
    if len(df) == 0:
        print("No data to plot.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df_sorted = df.sort_values("Date")

    plt.plot(df_sorted["Date"], df_sorted["Amount"])
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.title("Spending Over Time")
    plt.show()


# ------------------- PREDICTION -------------------
def predict_spending():
    if len(df) < 2:
        print("Not enough data to predict.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df_sorted = df.sort_values("Date")

    df_sorted["Date_ordinal"] = df_sorted["Date"].map(pd.Timestamp.toordinal)

    X = df_sorted["Date_ordinal"].values.reshape(-1, 1)
    y = df_sorted["Amount"].values

    model = LinearRegression()
    model.fit(X, y)

    next_date = np.array([[df_sorted["Date_ordinal"].max() + 1]])
    prediction = model.predict(next_date)

    print(f"\n🔮 Predicted next expense: {prediction[0]:.2f}")

# ------------------- MENU -------------------
while True:
    print("\n1. Add Expense")
    print("2. View Summary")
    print("3. Predict Next Expense")
    print("4. Monthly Summary")
    print("5. Show Graph")
    print("6. Exit")

    choice = input("Choose: ")

    if choice == "1":
        add_expense()

    elif choice == "2":
        view_summary()

    elif choice == "3":
        predict_spending()

    elif choice == "4":
        monthly_summary()

    elif choice == "5":
        show_graph()

    elif choice == "6":
        print("Goodbye 💖")
        break

    else:
        print("Invalid choice")