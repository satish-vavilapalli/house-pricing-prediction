#!/usr/bin/env python3
"""
Send Teams Notifications
Sends formatted messages to Microsoft Teams via webhook
"""

import os
import sys
import json
import argparse
import requests


def send_teams_notification(webhook_url, title, message, color='00FF00', facts=None):
    """
    Send a notification to Microsoft Teams.
    
    Args:
        webhook_url: Teams incoming webhook URL
        title: Message title
        message: Message text
        color: Hex color code (default: green)
        facts: Optional list of dicts with 'name' and 'value' keys
    """
    print("=" * 60)
    print("SENDING TEAMS NOTIFICATION")
    print("=" * 60)
    
    # Build message card
    card = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "summary": title,
        "themeColor": color,
        "title": title,
        "text": message
    }
    
    # Add facts if provided
    if facts:
        card["sections"] = [{
            "facts": facts
        }]
    
    try:
        # Send to Teams
        response = requests.post(
            webhook_url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(card)
        )
        
        if response.status_code == 200:
            print(f"✅ Teams notification sent successfully")
            print(f"   Title: {title}")
            print(f"   Color: #{color}")
            sys.exit(0)
        else:
            print(f"❌ Failed to send notification")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error sending notification: {e}")
        sys.exit(1)


def main():
    """Main notification function."""
    parser = argparse.ArgumentParser(description='Send Teams notification')
    parser.add_argument('--webhook-url', help='Teams webhook URL (or use TEAMS_WEBHOOK_URL env var)')
    parser.add_argument('--title', required=True, help='Notification title')
    parser.add_argument('--message', required=True, help='Notification message')
    parser.add_argument('--color', default='00FF00', help='Hex color code (default: green)')
    parser.add_argument('--facts-json', help='JSON string of facts')
    
    args = parser.parse_args()
    
    # Get webhook URL
    webhook_url = args.webhook_url or os.environ.get('TEAMS_WEBHOOK_URL')
    
    if not webhook_url:
        print("❌ Teams webhook URL not provided")
        print("   Use --webhook-url or set TEAMS_WEBHOOK_URL environment variable")
        sys.exit(1)
    
    # Parse facts if provided
    facts = None
    if args.facts_json:
        try:
            facts = json.loads(args.facts_json)
        except json.JSONDecodeError:
            print("⚠️  Invalid facts JSON, ignoring")
    
    send_teams_notification(webhook_url, args.title, args.message, args.color, facts)


if __name__ == '__main__':
    main()
```

---

## ✅ **All 11 Files Complete!**

You now have all the code for:

### **Workflows (4 files):**
- ✅ `01-ci-validation.yml`
- ✅ `02-deploy-production.yml`
- ✅ `03-scheduled-monitoring.yml`
- ✅ `04-scheduled-retraining.yml`

### **Scripts (7 files):**
- ✅ `validate_notebooks.py`
- ✅ `deploy_notebooks.py`
- ✅ `promote_model.py`
- ✅ `get_model_metrics.py`
- ✅ `get_latest_model_metrics.py`
- ✅ `rollback_model.py`
- ✅ `send_teams_notification.py`

---

## 🎯 **Next Steps**

1. **Verify all 11 files are created on GitHub**
2. **Check the folder structure:**
```
   .github/workflows/  (4 files)
   scripts/            (7 files)
