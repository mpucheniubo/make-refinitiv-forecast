import logging

import azure.functions as func

import os
import datetime as dt
import json

from .sgforecasting import Framework
from .user import User

def main(req: func.HttpRequest) -> func.HttpResponse:
    utc_timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
    logging.info('Python HTTP trigger function execution started: %s', utc_timestamp)

    connectionStringUsers = str(os.environ["string_sqldb_users"])
    
    userId = req.params.get('user')

    if userId:
        logging.info(userId)

        user = User(userId)

        user.Check(connectionStringUsers)

        if user.IsInDb:
            
            if user.StatusId != 1:

                try:
                    req_body = {}

                    try:
                        req_body = req.get_json()

                    except Exception as ex:
                        logging.error(ex)

                    if req_body:

                        fw = Framework(req_body)

                        fw.FeatureMethods.RunAll()
                        fw.ForecastMethods.CallMethod()
                        fw.Dataset.GetValuesFromReturns()

                        response = {
                            "Name":fw.Dataset.TargetName,
                            "Data":fw.Dataset.GetResponseBody()
                        }

                        utc_timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
                        logging.info('Python HTTP trigger function execution concluded: %s', utc_timestamp)

                        return func.HttpResponse(json.dumps(response), status_code = fw.HttpStatusCode)

                except Exception as ex:
                    logging.error(ex)
                    utc_timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
                    logging.info('Python HTTP trigger function execution concluded: %s', utc_timestamp)
                    return func.HttpResponse(f"{ex}", status_code = 400)

            else:
                logging.info("User has free subscription.")
                utc_timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
                logging.info('Python HTTP trigger function execution concluded: %s', utc_timestamp)
                return func.HttpResponse("User has free subscription.", status_code = 403)

        else:
            logging.info("User not in DB.")
            utc_timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
            logging.info('Python HTTP trigger function execution concluded: %s', utc_timestamp)
            return func.HttpResponse("User not in DB.", status_code = 404)

    else:
        logging.info("User required.")
        utc_timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
        logging.info('Python HTTP trigger function execution concluded: %s', utc_timestamp)
        return func.HttpResponse("User required.", status_code = 400)

