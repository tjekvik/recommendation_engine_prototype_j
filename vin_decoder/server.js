import { createDecoder } from "@cardog/corgi";
import http from "node:http";

const decoder = await createDecoder();

const statusForResult = (result) => {
  if (result.valid) {
    return 200;
  } else if (result.error) {
    return 400;
  } else {
    return 422;
  }
}

const createRequestHandler = (decoder) => {
return async (req, resp) => {
    try {
      if (req.method !== "GET") {
        return resp.writeHead(405).end();
      }
      const vin = req.url.slice(1);
      console.log("Decoding VIN:", vin);
      const result = await decoder.decode(vin);
      if (result.valid) {
        console.log("Decoded", vin, " successfully");
      } else {
        console.log("Failed to decode", vin, ":", result.errors || "unknown error");
      }

      resp.writeHead(statusForResult(result), {
        "content-type": "application/json",
      });
      const response= JSON.stringify(result);
      console .log("Response:", response);
      resp.end(response);
     
    } catch (e) {
      console.error(e);
      resp.writeHead(500).end();
    }
  };
};

http.createServer(createRequestHandler(decoder)).listen(4000 || process.env.PORT);
console.log("Listening on http://localhost:4000");

