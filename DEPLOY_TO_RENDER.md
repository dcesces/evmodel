# Deploying to Render

1. Create a GitHub repository and upload all files in this package.
2. Sign in to https://render.com
3. Click New + → Web Service.
4. Connect your GitHub repository.
5. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. In Environment Variables, add:
   - `TOMTOM_KEY` = your TomTom API key
7. Deploy.

Your app will be available at a stable public URL.
