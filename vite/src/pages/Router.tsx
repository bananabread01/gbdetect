import React, { Suspense } from 'react'
import {
    Route,
    createBrowserRouter,
    createRoutesFromElements,
    RouterProvider
} from 'react-router-dom';
import AppUI from './components/AppUI';
import LoadingPage from './Loading';
import Home from './Home';
import App from './App';
import Instructions from './Instructions';
import Results from './Results';
import Upload from './Upload';

const Router = () => {
    let router = createBrowserRouter(
        createRoutesFromElements(
            <Route path='/' element={<AppUI />} >
                <Route index element={<Home />} />
                <Route path='app' element={<App />} />
                <Route path='instructions' element={<Instructions />} />
                <Route path='upload' element={<Upload />} />
                <Route path='results' element={<Results />} />
            </Route>
        )
    )
    return (
        <Suspense fallback={<LoadingPage />}>
            <RouterProvider router={router} />
        </Suspense>
    )
}

export default Router