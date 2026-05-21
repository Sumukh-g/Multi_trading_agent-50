import pandas as pd, os
from jinja2 import Template
from datetime import datetime

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Quantitative Trading Signals & Portfolio</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      margin: 0;
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    .container {
      max-width: 1400px;
      margin: 0 auto;
      background: white;
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      overflow: hidden;
    }
    .header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 30px;
      text-align: center;
    }
    .header h1 {
      margin: 0 0 10px 0;
      font-size: 2.5em;
      font-weight: 700;
    }
    .header p {
      margin: 0;
      opacity: 0.9;
      font-size: 1.1em;
    }
    .stats {
      display: flex;
      justify-content: space-around;
      padding: 20px;
      background: #f8f9fa;
      border-bottom: 2px solid #e9ecef;
    }
    .stat-box {
      text-align: center;
    }
    .stat-value {
      font-size: 2em;
      font-weight: bold;
      color: #667eea;
    }
    .stat-label {
      color: #6c757d;
      font-size: 0.9em;
      margin-top: 5px;
    }
    .content {
      padding: 30px;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 20px 0;
      font-size: 0.9em;
    }
    th {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 12px 8px;
      text-align: right;
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    th:first-child {
      text-align: left;
    }
    td {
      padding: 10px 8px;
      text-align: right;
      border-bottom: 1px solid #e9ecef;
    }
    td:first-child {
      text-align: left;
      font-weight: 600;
    }
    tr:hover {
      background: #f8f9fa;
    }
    .actionable {
      background: #d4edda !important;
      border-left: 4px solid #28a745;
    }
    .unsure {
      background: #fff3cd !important;
      border-left: 4px solid #ffc107;
    }
    .long {
      color: #28a745;
      font-weight: bold;
    }
    .short {
      color: #dc3545;
      font-weight: bold;
    }
    .portfolio-section {
      margin-top: 40px;
      padding-top: 30px;
      border-top: 3px solid #e9ecef;
    }
    .portfolio-section h2 {
      color: #667eea;
      margin-bottom: 20px;
    }
    .portfolio-table {
      background: #f8f9fa;
      border-radius: 8px;
      overflow: hidden;
    }
    .portfolio-table th {
      background: #495057;
    }
    .weight-bar {
      background: #28a745;
      height: 20px;
      border-radius: 10px;
      display: inline-block;
      min-width: 30px;
      text-align: center;
      color: white;
      line-height: 20px;
      font-size: 0.85em;
      padding: 0 8px;
    }
    .footer {
      text-align: center;
      padding: 20px;
      color: #6c757d;
      font-size: 0.9em;
      border-top: 1px solid #e9ecef;
    }
    @media (max-width: 768px) {
      table { font-size: 0.75em; }
      th, td { padding: 6px 4px; }
      .header h1 { font-size: 1.8em; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>📊 Quantitative Trading Signals</h1>
      <p>AI-Powered Portfolio Analysis & Recommendations</p>
    </div>
    
    <div class="stats">
      <div class="stat-box">
        <div class="stat-value">{{ rows }}</div>
        <div class="stat-label">Total Signals</div>
      </div>
      <div class="stat-box">
        <div class="stat-value">{{ actionable }}</div>
        <div class="stat-label">Actionable Trades</div>
      </div>
      <div class="stat-box">
        <div class="stat-value">{{ (actionable / rows * 100) | round(1) if rows > 0 else 0 }}%</div>
        <div class="stat-label">Actionable Rate</div>
      </div>
    </div>
    
    <div class="content">
      <h2 style="color: #667eea; margin-top: 0;">Trading Signals</h2>
      <table>
        <thead>
          <tr>
            {% for c in cols %}<th>{{c.replace('_', ' ').title()}}</th>{% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for row in data %}
          <tr class="{{ 'actionable' if row['actionable'] else 'unsure' }}">
            {% for c in cols %}
            <td>
              {% if c == 'decision' %}
                <span class="{{ 'long' if row[c] == 'LONG' else 'short' if row[c] == 'SHORT' else '' }}">{{ row[c] }}</span>
              {% elif c == 'actionable' %}
                {{ '✅' if row[c] else '⏸️' }}
              {% elif c in ['prob_5pct_30d', 'confidence', 'exp_30d_return'] %}
                {{ '{:.2%}'.format(row[c]) if row[c] is not none else 'N/A' }}
              {% elif c in ['price', 'stop', 'target1', 'size'] %}
                ${{ '{:,.2f}'.format(row[c]) if row[c] is not none else 'N/A' }}
              {% else %}
                {{ row[c] if row[c] is not none else 'N/A' }}
              {% endif %}
            </td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
      
      {% if portfolio is not none and portfolio|length > 0 %}
      <div class="portfolio-section">
        <h2>💼 Optimized Portfolio Allocation</h2>
        <div class="portfolio-table">
          <table>
            <thead>
              <tr>
                <th style="text-align: left;">Ticker</th>
                <th>Weight</th>
                <th>Allocation</th>
              </tr>
            </thead>
            <tbody>
              {% for r in portfolio %}
              {% if r['weight'] > 0 %}
              <tr>
                <td style="text-align: left; font-weight: 600;">{{ r['ticker'] }}</td>
                <td>{{ '{:.2%}'.format(r['weight']) }}</td>
                <td>
                  <div class="weight-bar" style="width: {{ (r['weight'] * 500) | round }}px;">
                    {{ '{:.1%}'.format(r['weight']) }}
                  </div>
                </td>
              </tr>
              {% endif %}
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
      {% endif %}
    </div>
    
    <div class="footer">
      <p>Generated on {{ timestamp }} | Quantitative Trading Model v1.0</p>
    </div>
  </div>
</body>
</html>
"""

def main():
    if not os.path.exists("data/signals_latest.csv"):
        print("❌ Error: data/signals_latest.csv not found. Run signals generation first.")
        return
    
    sig = pd.read_csv("data/signals_latest.csv")
    os.makedirs("web", exist_ok=True)
    
    # Format columns for display
    cols = list(sig.columns)
    data = sig.to_dict(orient="records")
    
    portfolio = None
    if os.path.exists("data/portfolio_latest.csv"):
        pf = pd.read_csv("data/portfolio_latest.csv")
        if 'weight' in pf.columns and 'ticker' in pf.columns:
            portfolio = pf[pf['weight'] > 0][['ticker','weight']].to_dict(orient="records")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    tpl = Template(HTML)
    html = tpl.render(
        cols=cols, 
        data=data, 
        rows=len(sig), 
        actionable=int(sig['actionable'].sum()) if 'actionable' in sig.columns else 0,
        portfolio=portfolio,
        timestamp=timestamp
    )
    
    with open("web/signals.html","w",encoding="utf-8") as f:
        f.write(html)
    
    print("✅ Created web/signals.html")
    print(f"   📊 {len(sig)} signals, {int(sig['actionable'].sum()) if 'actionable' in sig.columns else 0} actionable")
    if portfolio:
        print(f"   💼 {len(portfolio)} positions in portfolio")
    print(f"   🌐 Open web/signals.html in your browser to view")

if __name__ == "__main__":
    main()
