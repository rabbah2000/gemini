import json
import google.generativeai as genai

# Replace with your actual API key
GOOGLE_API_KEY = 'XXXXXX'
genai.configure(api_key=GOOGLE_API_KEY)

def get_plages_par_ville(ville):
    try:
        # Load the JSON data
        with open('plage.json', 'r', encoding='utf-8') as f:
            plages = json.load(f)
        
        # Check if the city is in the JSON data
        if ville in plages:
            return plages[ville]
        else:
            return f"Ville '{ville}' non trouvée dans le fichier JSON."
    except FileNotFoundError:
        return "Fichier 'plages.json' non trouvé."
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON: {e}")
        return "Erreur de décodage du fichier JSON."
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        return "Une erreur inattendue s'est produite."

if __name__ == "__main__":
    ville = "Casablanca"
    plages = get_plages_par_ville(ville)
    print(plages)

    # Instantiate the generative model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    # Prepare the prompt for the model
    prompt = f"turn your JSON into something more engaging, give me a simple paragraph for every beach seperate:  {plages,ville}"

    # Generate the content
    response = model.generate_content(prompt)

    # Print the formatted response
    print(response.text)
