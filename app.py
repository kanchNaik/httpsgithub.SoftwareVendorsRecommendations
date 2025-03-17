from flask import Flask, request, jsonify
from VendorQualification.VendorQualifier import get_qualifiedVendors
from CommonProcessingUtility import get_query
app = Flask(__name__)

@app.route('/vendor_qualification', methods=['GET'])
def vendor_qualification ():
    
    """
    Endpoint to qualify vendors based on similarity to a provided query and filter by software category and capabilities.
    This function processes the incoming request, retrieves the list of qualified vendors, and returns them in a JSON format.

    Request Arguments:
        - software_category (str): The category of software the vendors must belong to.
        - capabilities (list): A list of capabilities used to refine the query (though not utilized in the current implementation).

    Returns:
        JSON Response:
            - 'message': A static message indicating the purpose of the endpoint ('Vendor Qualification').
            - 'similarity_scores': A list of top 10 qualified vendors, including product name, rating, seller, features, and similarity scores.
    """

    data = request.get_json()
    software_category = data.get('software_category', '').strip()
    capabilities = data.get('capabilities', [])

    query = get_query(software_category, capabilities)
    
    qualifiedVendors = get_qualifiedVendors("C:\\Users\\naikn\\Downloads\\G2 software product overview.csv", query, software_category, capabilities)
    qualifiedVendors_json = qualifiedVendors.to_dict(orient='records')
    
    return jsonify({
        'message': 'Vendor Qualification',
        'similarity_scores': qualifiedVendors_json
    })


if __name__ == '__main__':
    app.run(debug=True)
