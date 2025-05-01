#safety Stock

![](https://i.imgur.com/SKksFxz.png)

<h2>Description</h2>

Zeno is a conversational AI chatbot for businesses interact with theirdata. Zeno is designed to help you discover, explore, and manage your business reports with ease, empowering you to make data-driven decisions faster than ever before.

LLM Chatbot built using Next.js for front-end and backend-end and a self-hosted convex database

Features:
1. LLM Chat interface
2. Retrieval Augmented Generation
3. Authentication
4. About page
5. Reports page
6. Additional feature requests page


<h2>Programming Language</h2>

- Typescript & Javascript
- Html & CSS

<h2>Tech stack and CI/CD</h2>

- Next.js
- LangChain.js
- ConvexDB
- React.js
- Shadcn
- Zod
- Docker
- Ngnix
- Git

<h2>Environment Used </h2>

- <b>Ubuntu</b>

<h2>Run manually:</h2>

<p align="left">
 
1. Clone the project: Run this from the command line
 
 ```commandline
 git clone git@github.com:graphshade/zeno_convex.git
 ```
 
2. Change directory to zeno_convex and install the dependencies.
   
```commandline
 cd zeno_convex
 npm install
 ```

3. Set up the convex database. Follow instructions here to [self-host convex](https://stack.convex.dev/self-hosted-develop-and-deploy).
 
4. Update environment variables in .env.local file if needed

5. Update/generate the convex functions and run development server
   
```commandline
npx convex dev
npm run dev
 ```


<h2>Run with Docker:</h2>

<p align="left">
 
1. Clone the project: Run this from the command line
 
 ```commandline
 git clone git@github.com:graphshade/zeno_convex.git
 ```
 
2. Change directory to zeno_convex.
   
```commandline
 cd zeno_convex
 ```

3. Set up the convex database. Follow instructions here to [self-host convex](https://stack.convex.dev/self-hosted-develop-and-deploy).
 
4. Update environment variables in .env.local file if needed

5. Build docker image
   
```commandline
docker build -t zeno_convex $(xargs < .env.local echo -n --build-arg ) 
 ```
 
6. Run docker container

```commandline
docker run --env-file .env.local -p 3000:3000 zeno_convex
 ```

<h2>More Images </h2>

<p float="left">
  <img src="https://i.imgur.com/f9xxfki.png" width="500" />
  <img src="https://i.imgur.com/JFmmi7E.png" width="500" /> 
  <img src="https://i.imgur.com/Mx9hrXJ.png" width="500" />
  <img src="https://i.imgur.com/a5L6Sr6.png" width="500" />
</p>

